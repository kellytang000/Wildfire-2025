import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";
import jwt from "jsonwebtoken";

// 使用全局变量避免在开发环境中创建多个Prisma实例
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

const prisma = globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;

// GET - 获取当前用户位置
export async function GET(request: Request) {
  try {
    // 从请求头获取token
    const authHeader = request.headers.get('authorization');
    const token = authHeader?.replace('Bearer ', '');

    if (!token) {
      return NextResponse.json({ error: 'No token provided' }, { status: 401 });
    }

    // 验证token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key') as any;
    const userId = decoded.id;

    // 获取用户位置信息
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        username: true,
        lat: true,
        lng: true,
      },
    });

    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    return NextResponse.json({
      statusCode: 200,
      data: user,
    });
  } catch (error) {
    console.error('Error fetching user location:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// PUT - 更新当前用户位置
export async function PUT(request: Request) {
  try {
    // 从请求头获取token
    const authHeader = request.headers.get('authorization');
    const token = authHeader?.replace('Bearer ', '');

    if (!token) {
      return NextResponse.json({ error: 'No token provided' }, { status: 401 });
    }

    // 验证token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key') as any;
    const userId = decoded.id;

    // 获取请求体数据
    const { lat, lng } = await request.json();

    // 验证坐标数据
    if (typeof lat !== 'number' || typeof lng !== 'number') {
      return NextResponse.json({ error: 'Invalid coordinates' }, { status: 400 });
    }

    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      return NextResponse.json({ error: 'Coordinates out of range' }, { status: 400 });
    }

    // 更新用户位置
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: {
        lat: lat,
        lng: lng,
      },
      select: {
        id: true,
        username: true,
        lat: true,
        lng: true,
      },
    });

    return NextResponse.json({
      statusCode: 200,
      data: updatedUser,
      message: 'Location updated successfully',
    });
  } catch (error) {
    console.error('Error updating user location:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
