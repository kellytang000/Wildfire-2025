import { NextResponse } from "next/server";
import axios from "axios";
import * as cheerio from "cheerio";
import fs from "fs-extra";
import path from "path";
import { PYRECAST_BASE_URL } from "@/src/constant";

const DOWNLOAD_DIR = path.join(process.cwd(), "downloads");

export async function GET() {
  try {
    await crawlAndDownload(PYRECAST_BASE_URL);
    return NextResponse.json({ message: "爬虫完成 ✅" });
  } catch (err: any) {
    console.error("❌ 爬虫失败:", err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}

async function crawlAndDownload(url: string, relativePath = "") {
  console.log("📂 正在访问: ", url);
  const res = await axios.get(url);
  const $ = cheerio.load(res.data);

  const links = $("a")
    .map((_, el) => $(el).attr("href"))
    .get()
    .filter((href) => href && href !== "../");

  for (const link of links) {
    const isDir = link.endsWith("/");
    const fullUrl = new URL(link, url).href;
    const nextRelativePath = path.join(relativePath, link);

    if (isDir) {
      await crawlAndDownload(fullUrl, nextRelativePath);
    } else if (link.endsWith(".tif")) {
      const localPath = path.join(DOWNLOAD_DIR, relativePath, link);
      await downloadFile(fullUrl, localPath);
    }
  }
}

async function downloadFile(fileUrl: string, savePath: string) {
  try {
    await fs.ensureDir(path.dirname(savePath));
    const writer = fs.createWriteStream(savePath);
    const res = await axios.get(fileUrl, { responseType: "stream" });
    res.data.pipe(writer);

    await new Promise((resolve, reject) => {
      writer.on("finish", () => {
        resolve(null);
      });
      writer.on("error", reject);
    });
    console.log(`✅ 下载成功: ${fileUrl}`);
  } catch (err: any) {
    console.error(`❌ 下载失败: ${fileUrl}`, err.message);
  }
}
