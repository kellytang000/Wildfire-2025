import jwt from "jsonwebtoken";
import { NextRequest } from "next/server";
import { User } from "@prisma/client";

import { JWT_SECRET } from "../constant";

// 生成JWT
export function signJwt(payload: Record<string, any>, expiresIn = "24h"): string {
  return jwt.sign({ ...payload, exp: Math.floor(Date.now() / 1000) + 60 * 60 * 24 }, JWT_SECRET);
}

// 验证并解密JWT
export function verifyJwt<T = any>(token: string): T | null {
  return jwt.verify(token, JWT_SECRET) as T;
}

export const getUserId = (req: NextRequest): string | null => {
  const token = req.headers.get("authorization") || req.cookies.get("token")?.value;

  if (token) {
    const decoded = verifyJwt(token);

    return decoded ? decoded.id : null;
  }

  return null;
};

export const getUser = async (req: NextRequest): Promise<User | null> => {
  const token = req.headers.get("authorization") || req.cookies.get("token")?.value;

  if (token) {
    const decoded = verifyJwt(token);

    return decoded ? (decoded as User) : null;
  }

  return Promise.reject(null);
};
