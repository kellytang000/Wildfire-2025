// app/api/auth/verify/route.ts
import { NextResponse } from "next/server";

import { verifyJwt } from "@/src/utils/jwt";

export async function POST(req: Request) {
  const { token } = await req.json();

  try {
    const decoded = verifyJwt(token);

    return NextResponse.json({ valid: true, decoded });
  } catch (err) {
    return NextResponse.json({ valid: false }, { status: 401 });
  }
}
