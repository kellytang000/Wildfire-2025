import { fromUrl, fromArrayBuffer } from "geotiff";
import proj4 from "proj4";
import sharp from "sharp";

// 注册 EPSG:32612 → WGS84 的转换
proj4.defs("EPSG:32612", "+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs");
const utmToWgs84 = proj4("EPSG:32612", "EPSG:4326");

export async function convertTiffToPng(url: string, centerLat: number, centerLng: number) {
  const tiff = await fromUrl(url);

  console.log("下载成功");
  const image = await tiff.getImage();
  const [pixelWidth, pixelHeight] = image.getResolution();
  const width = image.getWidth();
  const height = image.getHeight();

  // 计算总米数
  const totalWidthMeters = width * Math.abs(pixelWidth);
  const totalHeightMeters = height * Math.abs(pixelHeight);

  // 米转度
  const latSpan = totalHeightMeters / 111000;
  const lngSpan = totalWidthMeters / (111000 * Math.cos((centerLat * Math.PI) / 180));

  // 计算边界
  const rectangle = {
    north: centerLat + latSpan / 2,
    south: centerLat - latSpan / 2,
    east: centerLng + lngSpan / 2,
    west: centerLng - lngSpan / 2,
  };

  console.log("转换后 WGS84 Bounds:", rectangle);

  const raster = (await image.readRasters({ interleave: true })) as any as number[];

  const rawData = raster; // 单通道灰度图

  // const resultImage = new Float32Array(width * height);
  const mask = new Uint8ClampedArray(width * height * 4);

  let minValue = Infinity;
  let maxValue = -Infinity;

  for (let i = 0; i < width * height; i++) {
    const v = rawData[i];

    // resultImage[i] = v;
    if (v !== 0) {
      minValue = Math.min(minValue, v);
      maxValue = Math.max(maxValue, v);
    }
  }

  // 映射到颜色（红色渐变）并写入 RGBA
  for (let i = 0; i < width * height; i++) {
    const v = rawData[i];

    if (v === 0 || isNaN(v)) {
      // 无值区域设为透明
      mask[i * 4 + 0] = 0; // R
      mask[i * 4 + 1] = 0; // G
      mask[i * 4 + 2] = 0; // B
      mask[i * 4 + 3] = 0; // A
      continue;
    }

    const t = (maxValue - v) / (maxValue - minValue); // 反向映射
    const red = Math.round(t * 255);
    const alpha = Math.round(t * 255); // 可调为 t * 200 更柔和

    mask[i * 4 + 0] = red;
    mask[i * 4 + 1] = 0;
    mask[i * 4 + 2] = 0;
    mask[i * 4 + 3] = alpha;
  }

  // // 创建 RGBA 图像 Buffer
  // const rgbaBuffer = Buffer.alloc(width * height * 4);

  // for (let i = 0; i < width * height; i++) {
  //   const value = rawData[i];

  //   // 设置 RGB
  //   rgbaBuffer[i * 4] = value; // R
  //   rgbaBuffer[i * 4 + 1] = value; // G
  //   rgbaBuffer[i * 4 + 2] = value; // B

  //   // 如果是黑色或白色，alpha = 0；其他 alpha = 255
  //   if (value === 0 || value === 255) {
  //     rgbaBuffer[i * 4 + 3] = 0;
  //   } else {
  //     rgbaBuffer[i * 4 + 3] = 255;
  //   }
  // }

  // 保存为 PNG
  const buffer = await sharp(mask, {
    raw: {
      width,
      height,
      channels: 4,
    },
  })
    .png()
    .toBuffer();

  console.log("✅ 转换完成：");

  return {
    rectangle,
    buffer,
    width,
    height,
    max: maxValue,
    min: minValue,
  };
}

function transformPixelToUTM(x: number, y: number, M: number[]) {
  return [M[0] + M[1] * x + M[2] * y, M[3] + M[4] * x + M[5] * y];
}

export const transformTiffToPng = async (buf: ArrayBuffer) => {
  // Load our data tile from url, arraybuffer, or blob, so we can work with it:
  const tiff = await fromArrayBuffer(buf);
  const image = await tiff.getImage(); // by default, the first image is read.

  // Construct the WGS-84 forward and inverse affine matrices:
  const { ModelPixelScale: s, ModelTiepoint: t } = image.fileDirectory;
  let [sx, sy] = s;
  let [, , , gx, gy] = t; // tiepoint 是像素坐标 (0,0) 对应的地图坐标 (gx, gy)

  sy = -sy; // 像素Y方向翻转

  // 仿射矩阵：像素 -> UTM 坐标（米）
  const pixelToUTM = [gx, sx, 0, gy, 0, sy];

  const width = image.getWidth();
  const height = image.getHeight();

  // 四个角的像素坐标
  const corners = [
    [0, 0], // 左上
    [width, 0], // 右上
    [width, height], // 右下
    [0, height], // 左下
  ];

  // 转换为 WGS84 经纬度
  const cornerLatLngs = corners.map(([x, y]) => {
    const [utmX, utmY] = transformPixelToUTM(x, y, pixelToUTM);

    return utmToWgs84.forward([utmX, utmY]); // 返回 [lon, lat]
  });

  // 提取经纬度边界值
  const lons = cornerLatLngs.map(([lon]) => lon);
  const lats = cornerLatLngs.map(([, lat]) => lat);

  const bounds = {
    west: Math.min(...lons),
    east: Math.max(...lons),
    south: Math.min(...lats),
    north: Math.max(...lats),
  };

  const raster = (await image.readRasters({ interleave: true })) as any as number[];

  const rawData = raster;

  // 创建 RGBA 图像 Buffer
  const rgbaBuffer = Buffer.alloc(width * height * 4);

  for (let i = 0; i < width * height; i++) {
    const value = rawData[i];

    // 设置 RGB
    rgbaBuffer[i * 4] = value; // R
    rgbaBuffer[i * 4 + 1] = value; // G
    rgbaBuffer[i * 4 + 2] = value; // B

    // 如果是黑色或白色，alpha = 0；其他 alpha = 255
    if (value === 0 || value === 255) {
      rgbaBuffer[i * 4 + 3] = 0;
    } else {
      rgbaBuffer[i * 4 + 3] = 255;
    }
  }

  // 保存为 PNG
  const buffer = await sharp(rgbaBuffer, {
    raw: {
      width,
      height,
      channels: 4,
    },
  })
    .png()
    .toBuffer();

  console.log("✅ 转换完成：");

  return {
    rectangle: bounds,
    buffer,
    width,
    height,
  };
};
