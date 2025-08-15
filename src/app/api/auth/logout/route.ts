import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";

export const GET = async (req: NextRequest) => {
  const ck = await cookies();

  ck.delete("token");

  return NextResponse.json({
    statusCode: 200,
  });
};
