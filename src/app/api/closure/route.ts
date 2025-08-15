import { NextRequest, NextResponse } from "next/server";

import { $db } from "@/src/db/prisma";
import { getUserId } from "@/src/utils/jwt";

export const GET = async () => {
  const markers = await $db.roadClosure.findMany({ include: { User: true } });

  return NextResponse.json({ data: markers, statusCode: 200 });
};

export const POST = async (req: NextRequest) => {
  const userId = getUserId(req);
  const body = await req.json();

  const marker = await $db.roadClosure.create({ data: { ...body, userId } });

  return NextResponse.json({ data: marker, statusCode: 200 });
};

export const DELETE = async (req: NextRequest) => {
  const id = getUserId(req);
  const userId = id ? Number(id) : null;
  const body = await req.json();

  if (userId) {
    const marker = await $db.roadClosure.delete({ where: { id: body.id } });

    return NextResponse.json({ data: marker, statusCode: 200 });
  }
};
