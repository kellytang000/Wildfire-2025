import { NextRequest, NextResponse } from "next/server";

import { getUser } from "@/src/utils/jwt";
import { $db } from "@/src/db/prisma";

export const GET = async (req: NextRequest) => {
  try {
    const user = await getUser(req);

    if (user) {
      const curUser = await $db.user.findUnique({ where: { id: user.id }, omit: { password: true } });

      if (curUser) {
        return NextResponse.json({ data: curUser, statusCode: 200 });
      }

      return NextResponse.json({ statusCode: 401 });
    }

    return NextResponse.json({ data: user, statusCode: 401 });
  } catch (error) {
    return NextResponse.json({ statusCode: 401 });
  }
};
