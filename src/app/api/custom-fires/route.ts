import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

// 使用全局变量避免在开发环境中创建多个Prisma实例
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;

export interface CustomFireFeature {
  type: "Feature";
  properties: {
    OBJECTID: number;
    type: string;
    Shape__Area: number;
    Shape__Length: number;
  };
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
}

export interface CustomFirePoint {
  id: number;
  lat: number;
  lng: number;
  properties: {
    OBJECTID: number;
    type: string;
    Shape__Area: number;
    Shape__Length: number;
  };
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
}

export const GET = async () => {
  try {
    // 从数据库获取自定义火灾数据
    const customFires = await prisma.customFire.findMany({
      orderBy: {
        id: 'asc'
      }
    });

    // 转换为前端需要的格式
    const customFirePoints: CustomFirePoint[] = customFires.map((fire) => {
      return {
        id: fire.id,
        lat: fire.lat,
        lng: fire.lng,
        properties: {
          OBJECTID: Number(fire.objectId), // Convert BigInt to Number
          type: fire.type,
          Shape__Area: fire.shapeArea,
          Shape__Length: fire.shapeLength,
        },
        geometry: JSON.parse(fire.geometry),
      };
    });

    return NextResponse.json({
      data: customFirePoints,
      statusCode: 200
    });
  } catch (error) {
    console.error("Error fetching custom fires from database:", error);
    return NextResponse.json({
      error: "Failed to load custom fires",
      statusCode: 500
    }, { status: 500 });
  }
};

export const POST = async (request: Request) => {
  try {
    const body = await request.json();
    const { lat, lng, type, geometry } = body;

    if (!lat || !lng || !type || !geometry) {
      return NextResponse.json({
        error: "Missing required fields: lat, lng, type, geometry",
        statusCode: 400
      }, { status: 400 });
    }

    // 计算多边形面积和周长（简化版本）
    const coordinates = geometry.coordinates[0];
    let area = 0;
    let perimeter = 0;

    for (let i = 0; i < coordinates.length - 1; i++) {
      const [x1, y1] = coordinates[i];
      const [x2, y2] = coordinates[i + 1];
      area += (x1 * y2 - x2 * y1);
      perimeter += Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }
    area = Math.abs(area) / 2;

    // 创建新的自定义火灾记录
    const newCustomFire = await prisma.customFire.create({
      data: {
        objectId: Date.now(), // 使用时间戳作为唯一ID
        lat: parseFloat(lat),
        lng: parseFloat(lng),
        type: type,
        shapeArea: area,
        shapeLength: perimeter,
        geometry: JSON.stringify(geometry),
      },
    });

    return NextResponse.json({
      data: {
        id: newCustomFire.id,
        lat: newCustomFire.lat,
        lng: newCustomFire.lng,
        properties: {
          OBJECTID: Number(newCustomFire.objectId), // Convert BigInt to Number
          type: newCustomFire.type,
          Shape__Area: newCustomFire.shapeArea,
          Shape__Length: newCustomFire.shapeLength,
        },
        geometry: JSON.parse(newCustomFire.geometry),
      },
      statusCode: 201
    });
  } catch (error) {
    console.error("Error creating custom fire:", error);
    return NextResponse.json({
      error: "Failed to create custom fire",
      statusCode: 500
    }, { status: 500 });
  }
};

export const DELETE = async (request: Request) => {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json({
        error: "Missing fire ID",
        statusCode: 400
      }, { status: 400 });
    }

    await prisma.customFire.delete({
      where: {
        id: parseInt(id)
      }
    });

    return NextResponse.json({
      message: "Custom fire deleted successfully",
      statusCode: 200
    });
  } catch (error) {
    console.error("Error deleting custom fire:", error);
    return NextResponse.json({
      error: "Failed to delete custom fire",
      statusCode: 500
    }, { status: 500 });
  }
};

export const PUT = async (request: Request) => {
  try {
    const body = await request.json();
    const { id, lat, lng, type, geometry } = body;

    if (!id || !lat || !lng || !type || !geometry) {
      return NextResponse.json({
        error: "Missing required fields: id, lat, lng, type, geometry",
        statusCode: 400
      }, { status: 400 });
    }

    // 计算多边形面积和周长（简化版本）
    const coordinates = geometry.coordinates[0];
    let area = 0;
    let perimeter = 0;

    for (let i = 0; i < coordinates.length - 1; i++) {
      const [x1, y1] = coordinates[i];
      const [x2, y2] = coordinates[i + 1];
      area += (x1 * y2 - x2 * y1);
      perimeter += Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }
    area = Math.abs(area) / 2;

    // 更新自定义火灾记录
    const updatedCustomFire = await prisma.customFire.update({
      where: {
        id: parseInt(id)
      },
      data: {
        lat: parseFloat(lat),
        lng: parseFloat(lng),
        type: type,
        shapeArea: area,
        shapeLength: perimeter,
        geometry: JSON.stringify(geometry),
      },
    });

    return NextResponse.json({
      data: {
        id: updatedCustomFire.id,
        lat: updatedCustomFire.lat,
        lng: updatedCustomFire.lng,
        properties: {
          OBJECTID: updatedCustomFire.objectId,
          type: updatedCustomFire.type,
          Shape__Area: updatedCustomFire.shapeArea,
          Shape__Length: updatedCustomFire.shapeLength,
        },
        geometry: JSON.parse(updatedCustomFire.geometry),
      },
      statusCode: 200
    });
  } catch (error) {
    console.error("Error updating custom fire:", error);
    return NextResponse.json({
      error: "Failed to update custom fire",
      statusCode: 500
    }, { status: 500 });
  }
};
