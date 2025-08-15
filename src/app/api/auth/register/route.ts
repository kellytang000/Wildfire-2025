import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { $db } from "@/src/db/prisma";
import { signJwt } from "@/src/utils/jwt";

export const POST = async (req: Request) => {
  const { username, password, role } = await req.json();
  const existingUser = await $db.user.findFirst({
    where: {
      username,
    },
  });

  if (existingUser) {
    return NextResponse.json({
      statusCode: 409,
      message: "User already exists",
    });
  }

  // Call your register function here
  const user = await $db.user.create({
    data: {
      username,
      password,
      role,
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
      statusCode: 400,
      message: "Registration failed, please try again",
    });
  }
};
