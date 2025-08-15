import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { $db } from "@/src/db/prisma";
import { signJwt } from "@/src/utils/jwt";

export const POST = async (req: Request) => {
  const { username, password } = await req.json();

  // Call your login function here
  const user = await $db.user.findFirst({
    where: {
      username,
      password,
    },
    omit: {
      password: true,
    },
  });

  if (user) {
    const token = signJwt({ id: user.id, username });
    const ck = await cookies();

    ck.set("token", token);

    return NextResponse.json(
      {
        statusCode: 200,
        data: user,
      },
      {
        headers: {
          Authorization: token,
        },
      },
    );
  } else {
    return NextResponse.json({
      statusCode: 401,
      message: "Invalid credentials",
    });
  }
};
