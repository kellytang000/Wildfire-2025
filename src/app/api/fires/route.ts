import axios from "axios";
import { NextRequest, NextResponse } from "next/server";
import Papa from "papaparse";

import { FIRES_URL } from "@/src/constant";
import { FirePoint } from "@/src/actions/maker.hook";

export const GET = async (req: NextRequest) => {
  const res = await axios.get(FIRES_URL);
  const text = res.data as string;  const markers = await new Promise((resolve) => {
    Papa.parse(text, {
      header: false, // 远程文件没有头部
      skipEmptyLines: true,
      complete: (results) => {
        const totalRows = results.data.length;
        console.log(`Total rows in CSV: ${totalRows}`);
        
        const parsed: FirePoint[] = (results.data as any[])
          .map((row: any[], index) => {
            // Skip empty rows
            if (!row || row.length < 4) {
              return null;
            }
            
            // CSV格式: name,date,longitude,latitude,prettyname,acres,containper...
            const name = row[0];
            const date = row[1];
            const longitude = row[2];
            const latitude = row[3];
            
            // Debug log for first few rows
            if (index < 3) {
              console.log(`Row ${index}:`, { name, date, longitude, latitude });
            }
            
            const lat = parseFloat(latitude);
            const lng = parseFloat(longitude);
            
            // Log for debugging
            if (isNaN(lat) || isNaN(lng)) {
              if (index < 3) {
                console.log('Invalid coordinates found:', { 
                  latitude, 
                  longitude, 
                  parsedLat: lat, 
                  parsedLng: lng 
                });
              }
              return null;
            }
            
            // Only include markers with valid coordinates
            if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
              console.log('Out of range coordinates:', { lat, lng });
              return null;
            }
            
            // Create raw object with proper field names for compatibility
            const rawObject = {
              name: row[0],
              date: row[1],
              longitude: row[2],
              latitude: row[3],
              prettyname: row[4],
              acres: row[5],
              containper: row[6],
              xlo: row[7],
              xhi: row[8],
              ylo: row[9],
              yhi: row[10],
              irwinid: row[11],
              htb: row[12],
              modis: row[13],
              viirs: row[14],
              wfigs: row[15],
              firis: row[16],
              'fireguard ': row[17]
            };
            
            return {
              lat,
              lng,
              raw: rawObject,
            };
          })
          .filter((marker): marker is FirePoint => marker !== null);

        console.log(`Valid markers after filtering: ${parsed.length} out of ${totalRows}`);
        resolve(parsed);
      },
    });
  });

  return NextResponse.json({ data: markers, statusCode: 200 });
};
