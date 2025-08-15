import { NextRequest, NextResponse } from "next/server";
import { $db } from "@/src/db/prisma";

export interface ShelterData {
  id: number;
  shelterId: string;
  lat: number;
  lng: number;
  capacity: number;
  hexId: string;
  region: string;
  createdAt: Date;
  updatedAt: Date;
}

export const GET = async (req: NextRequest) => {
  try {
    const { searchParams } = new URL(req.url);
    const region = searchParams.get('region');

    // Build query conditions
    const whereClause = region ? { region } : {};

    const shelters = await $db.shelter.findMany({
      where: whereClause,
      orderBy: [
        { region: 'asc' },
        { shelterId: 'asc' }
      ]
    });

    return NextResponse.json({ 
      data: shelters, 
      statusCode: 200 
    });
  } catch (error) {
    console.error("Error fetching shelters:", error);
    return NextResponse.json({ 
      error: "Failed to fetch shelters", 
      statusCode: 500 
    }, { status: 500 });
  }
};

export const POST = async (req: NextRequest) => {
  try {
    const body = await req.json();
    const { shelterId, lat, lng, capacity, hexId, region } = body;

    // Validate required fields
    if (!shelterId || !lat || !lng || !capacity || !hexId || !region) {
      return NextResponse.json({ 
        error: "Missing required fields", 
        statusCode: 400 
      }, { status: 400 });
    }

    const shelter = await $db.shelter.create({
      data: {
        shelterId,
        lat: parseFloat(lat),
        lng: parseFloat(lng),
        capacity: parseInt(capacity),
        hexId,
        region
      }
    });

    return NextResponse.json({ 
      data: shelter, 
      statusCode: 201 
    });
  } catch (error) {
    console.error("Error creating shelter:", error);
    return NextResponse.json({ 
      error: "Failed to create shelter", 
      statusCode: 500 
    }, { status: 500 });
  }
};

export const PUT = async (req: NextRequest) => {
  try {
    const body = await req.json();
    const { id, ...updateData } = body;

    if (!id) {
      return NextResponse.json({ 
        error: "Shelter ID is required", 
        statusCode: 400 
      }, { status: 400 });
    }

    const shelter = await $db.shelter.update({
      where: { id: parseInt(id) },
      data: {
        ...updateData,
        updatedAt: new Date()
      }
    });

    return NextResponse.json({ 
      data: shelter, 
      statusCode: 200 
    });
  } catch (error) {
    console.error("Error updating shelter:", error);
    return NextResponse.json({ 
      error: "Failed to update shelter", 
      statusCode: 500 
    }, { status: 500 });
  }
};

export const DELETE = async (req: NextRequest) => {
  try {
    const { searchParams } = new URL(req.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json({ 
        error: "Shelter ID is required", 
        statusCode: 400 
      }, { status: 400 });
    }

    const shelter = await $db.shelter.delete({
      where: { id: parseInt(id) }
    });

    return NextResponse.json({ 
      data: shelter, 
      statusCode: 200 
    });
  } catch (error) {
    console.error("Error deleting shelter:", error);
    return NextResponse.json({ 
      error: "Failed to delete shelter", 
      statusCode: 500 
    }, { status: 500 });
  }
};
