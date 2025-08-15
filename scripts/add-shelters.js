// Script to add 8 new shelter locations directly to database
const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

const shelterData = [
  // 你提到的两个地点
  {
    shelterId: "shelter_vancouver_01",
    lat: 49.7006,
    lng: -123.1556,
    capacity: 150,
    hexId: "hex_vancouver_01", 
    region: "vancouver"
  },
  {
    shelterId: "shelter_vancouver_02", 
    lat: 49.7822941,
    lng: -123.1584375,
    capacity: 200,
    hexId: "hex_vancouver_02",
    region: "vancouver"
  },
  // Spokane地区的两个地点
  {
    shelterId: "shelter_spokane_01",
    lat: 47.6587,
    lng: -117.4260,
    capacity: 180,
    hexId: "hex_spokane_01",
    region: "spokane"
  },
  {
    shelterId: "shelter_spokane_02",
    lat: 47.6398,
    lng: -117.4265,
    capacity: 120,
    hexId: "hex_spokane_02", 
    region: "spokane"
  },
  // Denver地区的两个地点
  {
    shelterId: "shelter_denver_01",
    lat: 39.7392,
    lng: -104.9903,
    capacity: 300,
    hexId: "hex_denver_01",
    region: "denver"
  },
  {
    shelterId: "shelter_denver_02", 
    lat: 39.7817,
    lng: -105.0178,
    capacity: 250,
    hexId: "hex_denver_02",
    region: "denver"
  },
  // LA地区的两个地点
  {
    shelterId: "shelter_la_01",
    lat: 34.0522,
    lng: -118.2437,
    capacity: 400,
    hexId: "hex_la_01",
    region: "la"
  },
  {
    shelterId: "shelter_la_02",
    lat: 34.0928, 
    lng: -118.3287,
    capacity: 350,
    hexId: "hex_la_02",
    region: "la"
  }
];

// 批量创建shelter的函数
async function addShelters() {
  try {
    console.log('🚀 Starting to add shelters...');
    
    for (const shelter of shelterData) {
      try {
        const result = await prisma.shelter.create({
          data: shelter
        });
        console.log(`✅ Added shelter: ${shelter.shelterId} (ID: ${result.id})`);
      } catch (error) {
        if (error.code === 'P2002') {
          console.log(`⚠️  Shelter ${shelter.shelterId} already exists, skipping...`);
        } else {
          console.error(`❌ Failed to add shelter ${shelter.shelterId}:`, error.message);
        }
      }
    }
  } catch (error) {
    console.error('❌ Error in addShelters:', error);
  } finally {
    await prisma.$disconnect();
  }
}

// 执行添加
addShelters().then(() => {
  console.log('🎉 Finished adding shelters');
}).catch((error) => {
  console.error('❌ Script failed:', error);
  process.exit(1);
});
