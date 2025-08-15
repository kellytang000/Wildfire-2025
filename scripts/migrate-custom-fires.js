const { PrismaClient } = require('@prisma/client');
const fs = require('fs');
const path = require('path');

const prisma = new PrismaClient();

async function migrateCustomFires() {
  try {
    console.log('Starting custom fires migration...');
    
    // Read the GeoJSON file
    const geoJsonPath = path.join(__dirname, '../src/script/squa/outputs/fire_combined.geojson');
    const geoJsonData = JSON.parse(fs.readFileSync(geoJsonPath, 'utf8'));
    
    console.log(`Found ${geoJsonData.features.length} custom fires in GeoJSON`);
    
    // Clear existing data
    await prisma.customFire.deleteMany();
    console.log('Cleared existing custom fire data');
    
    // Process each feature
    for (const feature of geoJsonData.features) {
      const { properties, geometry } = feature;
      
      // Calculate center point from polygon coordinates
      const coordinates = geometry.coordinates[0]; // First ring of polygon
      let latSum = 0;
      let lngSum = 0;
      
      for (const coord of coordinates) {
        lngSum += coord[0];
        latSum += coord[1];
      }
      
      const centerLat = latSum / coordinates.length;
      const centerLng = lngSum / coordinates.length;
      
      // Create database record
      await prisma.customFire.create({
        data: {
          objectId: properties.OBJECTID,
          lat: centerLat,
          lng: centerLng,
          type: properties.type || 'Heat Perimeter',
          shapeArea: properties.Shape__Area || 0,
          shapeLength: properties.Shape__Length || 0,
          geometry: JSON.stringify(geometry),
        },
      });
      
      console.log(`Migrated custom fire ${properties.OBJECTID}`);
    }
    
    console.log('Migration completed successfully!');
    
    // Verify the data
    const count = await prisma.customFire.count();
    console.log(`Total custom fires in database: ${count}`);
    
  } catch (error) {
    console.error('Migration failed:', error);
  } finally {
    await prisma.$disconnect();
  }
}

migrateCustomFires();
