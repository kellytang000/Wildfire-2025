import { exec } from "child_process";
import { join } from "path";

import { NextResponse } from "next/server";

import { SCRIPT_PATH } from "@/src/constant";

export const POST = async (req: Request) => {
  try {
    const { latitude, longitude } = await req.json();
    console.log(`🚀 路径规划请求: lat=${latitude}, lng=${longitude}`);

    const command = `python3 ${join(SCRIPT_PATH, "evac_la_squamish.py")} ${latitude} ${longitude} --json`;
    console.log(`📝 执行命令: ${command}`);

    const geojson = await new Promise<string>((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          console.error(`❌ Python脚本执行错误:`, error);
          console.error(`❌ stderr:`, stderr);
          reject(error);
          return;
        }

        if (stderr) {
          console.warn(`⚠️ Python脚本警告:`, stderr);
        }

        console.log(`✅ Python脚本输出:`, stdout.substring(0, 200) + "...");
        resolve(stdout);
      });
    });

    const parsedData = JSON.parse(geojson);
    console.log(`✅ 路径规划成功: ${parsedData.n_points} 点, ${parsedData.cost_s}秒`);

    return NextResponse.json({
      data: parsedData,
      statusCode: 200,
    });
  } catch (error) {
    console.error(`❌ API错误:`, error);
    return NextResponse.json(
      {
        error: "路径规划失败",
        details: error instanceof Error ? error.message : String(error),
        statusCode: 500,
      },
      { status: 500 },
    );
  }
};
