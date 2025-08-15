import { NextRequest, NextResponse } from "next/server";

import { getUserId } from "@/src/utils/jwt";
import { $db } from "@/src/db/prisma";

export const POST = async (req: NextRequest) => {
  const { latitude, longitude } = await req.json();
  const userId = getUserId(req);

  if (userId) {
    const data = await $db.reportWildfire.create({
      data: {
        lat: parseFloat(latitude),
        lng: parseFloat(longitude),
        userId: Number(userId!),
      },
    });

    return NextResponse.json({
      data,
      statusCode: 200,
    });
  }
};

export const GET = async () => {
  const data = await $db.reportWildfire.findMany();

  console.log(data, "2222222222222");

  return NextResponse.json({
    data,
    statusCode: 200,
  });
};

export const DELETE = async (req: NextRequest) => {
  const { id } = await req.json();

  if (id) {
    await $db.reportWildfire.delete({
      where: {
        id: Number(id),
      },
    });

    return NextResponse.json({
      statusCode: 200,
    });
  }

  return NextResponse.json({
    statusCode: 400,
  });
};
