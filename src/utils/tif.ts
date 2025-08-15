import { Math, Rectangle, SingleTileImageryProvider, Viewer } from 'cesium';
import { API } from './http';

export async function loadTiffAndRender(viewer: Viewer, tifUrl: string) {
  // const response = await fetch(tifUrl,{credentials:"include"});
  const {data,width,height,bbox} = await API.get<Record<string, any>>("/tif", { data: { url: tifUrl } });
  // const url = URL.createObjectURL(new Blob([data], { type: "image/png" }));
  // 用 Cesium 加载该图层
  const provider = new SingleTileImageryProvider({
    rectangle: Rectangle.fromDegrees(bbox.west, bbox.south, bbox.east, bbox.north),
    tileWidth: width,
    tileHeight: height,
    url:data,
  });

  viewer.imageryLayers.addImageryProvider(provider);

}
