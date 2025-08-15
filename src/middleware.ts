// middleware.ts
import { NextResponse, type NextRequest } from "next/server";

// 可配置：哪些路径不需要鉴权
const PUBLIC_PATHS = ["/auth/signin", "/register", "/api/public", "/auth/logout"];

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get("token")?.value;

  // 1. 跳过公开路径
  const skip = PUBLIC_PATHS.some((path) => pathname.startsWith(path));

  const verify = async () => {
    const res = await fetch(new URL("/api/auth/verify", request.url), {
      method: "POST",
      body: JSON.stringify({ token }),
      headers: { "Content-Type": "application/json" },
    });

    return res.json();
  };

  console.log(pathname, skip, token, "pathname-----------------");
  if (skip) {
    if (token) {
      try {
        // 3. 验证 JWT
        const json = await verify();

        request.headers.set("user", json.decoded.id);

        return NextResponse.redirect("/"); // 通过验证
      } catch (error) {
        return redirectToLogin(request);
      }
    } else {
      return NextResponse.next();
    }
  }
  // 2. 获取 Authorization Header

  if (!token) {
    return redirectToLogin(request);
  }

  try {
    // 3. 验证 JWT
    const json = await verify();

    request.headers.set("user", json.decoded.id);

    return NextResponse.next(); // 通过验证
  } catch (error) {
    console.error("JWT verification failed:", error);

    return redirectToLogin(request);
  }
}

// 4. 未通过验证，跳转到 /login
function redirectToLogin(request: NextRequest) {
  const loginUrl = new URL("/auth/signin", request.url);

  loginUrl.searchParams.set("callbackUrl", request.nextUrl.pathname); // 可选：回调原路径

  return NextResponse.redirect(loginUrl);
}

// 5. 配置作用范围 匹配所有请求
export const config = {
  // matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'], // 匹配所有页面，排除静态资源
  matcher: ["/((?!api|_next/static|_next/image|.*\\.png$|.*\\.jpg$|.*\\.mp4$|.*\\.jpeg$|.*\\.ico$).*)"],
};

export const runtime = "nodejs";
