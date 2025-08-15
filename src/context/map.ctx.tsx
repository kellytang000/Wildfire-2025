"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, Fragment } from "react";
import { GoogleMap, InfoWindow, LoadScript, Marker, GroundOverlay, Polyline, Polygon } from "@react-google-maps/api";
import { Button, Popconfirm, Table } from "antd";
import { useBoolean } from "ahooks";
import { Actions } from "ahooks/lib/useBoolean";
import dayjs from "dayjs";
import { createRoot } from "react-dom/client";

import { FirePoint, IRoadClosure, useClosureMarkers, useMapMarkers, useReportMarkers } from "../actions/maker.hook";
import { OverlayData } from "../components/sidebar/Left";
import { CustomFirePoint, useCustomFires } from "../actions/custom-fires.hook";
import { ShelterData, useShelters } from "../app/(home)/shelter/_hook/shelters.hook";
import { useCurrentUserLocation } from "../actions/user-location.hook";
import MapControls from "../components/MapControls";
type Point = { lat: number; lng: number };

export interface IRoutePath {
  route: number[][];
  cost_s: number;
  n_points: number;
  start: number[];
  destination: number[];
}

// Map theme types
export type MapTheme = 'darkview' | 'original';

type MapData = {
  loaded: boolean;
  setCenter: (center: Point) => void;
  currentMarker: FirePoint | null;
  setCurrentMarker: (marker: FirePoint | null) => void;
  map: google.maps.Map;
  setOverlayData: (data: OverlayData | null) => void;
  closureMode: boolean;
  setClosureMode: Actions;
  route: IRoutePath | null;
  setRoute: (route: IRoutePath | null) => void;
  setTifUrl: (url: string | null) => void;

  currentCustomFire: CustomFirePoint | null;
  setCurrentCustomFire: (fire: CustomFirePoint | null) => void;
  currentShelter: ShelterData | null;
  setCurrentShelter: (shelter: ShelterData | null) => void;
  showShelters: boolean;
  setShowShelters: (show: boolean) => void;
  refreshShelters: () => void;
  
  // Map theme properties
  mapTheme: MapTheme;
  setMapTheme: (theme: MapTheme) => void;
  mapType: 'roadmap' | 'satellite' | 'hybrid' | 'terrain';
  setMapType: (type: 'roadmap' | 'satellite' | 'hybrid' | 'terrain') => void;
  
  // User location control
  useUserLocation: boolean;
  setUseUserLocation: (use: boolean) => void;
  resetToDefaultCenter: () => void;
  
  // Simulated user location for address switching
  simulatedUserLocation: {lat: number, lng: number, address: string} | null;
  setSimulatedUserLocation: (location: {lat: number, lng: number, address: string} | null) => void;
};
const containerStyle = {
  width: "100%",
  height: "100vh",
  marginLeft: "var(--sidebar-width)",
  position: "fixed" as const,
  top: 0,
  right: 0,
};

const MapContext = createContext<MapData | null>(null);

let overlayWindow: google.maps.InfoWindow;

const excludedKeys = ["xlo", "xhi", "ylo", "yhi", "irwinid", "htb"];

const getUnit = (tifUrl: string) => {
  if (tifUrl.indexOf("flame-length") > -1) {
    return "ft";
  } else if (tifUrl.indexOf("hours-since-burned") > -1) {
    return "hours";
  } else if (tifUrl.indexOf("spread-rate") > -1) {
    return "ft/min";
  } else {
    return "s";
  }
};

export const MapProvider = ({ children }: { children: React.ReactNode }) => {
  // 默认地图中心点（丹佛）- 保持固定不变
  const DEFAULT_CENTER = { lat: 39.742043, lng: -104.991531 };
  const [center, setCenter] = useState(DEFAULT_CENTER); 
  // 模拟用户位置状态 - 用于在地图上显示用户选择的地址位置
  const [simulatedUserLocation, setSimulatedUserLocation] = useState<{lat: number, lng: number, address: string} | null>({
    lat: 49.78888350, 
    lng: -123.1318467, 
    address: 'Alice Lake Rd, Brackendale, BC'
  });
  const [useUserLocation, setUseUserLocation] = useState(false); // 始终为 false，不使用用户位置
  const { markers } = useMapMarkers();
  const { reportMarkers, delReport } = useReportMarkers();
  const { customFires } = useCustomFires();
  const { currentLocation } = useCurrentUserLocation();
  const [currentMarker, setCurrentMarker] = useState<FirePoint | null>(null);
  const [userLocationKey, setUserLocationKey] = useState(0);

  // Helper function to check if coordinates are valid
  const isValidCoordinate = (lat: number | null, lng: number | null): boolean => {
    return typeof lat === 'number' && 
           typeof lng === 'number' && 
           !isNaN(lat) && 
           !isNaN(lng) && 
           lat >= -90 && 
           lat <= 90 && 
           lng >= -180 && 
           lng <= 180;
  };

  // User location effect - disabled to keep map centered on Denver
  useEffect(() => {
    // This effect is intentionally disabled to keep map centered on Denver
    // Original logic: if (useUserLocation && currentLocation && isValidCoordinate(...))
    console.log('User location effect disabled - map stays at Denver');
  }, [currentLocation, useUserLocation]);
  
  // Helper function to reset to default center  
  const resetToDefaultCenter = () => {
    setCenter(DEFAULT_CENTER);
    setUseUserLocation(false);
  };
  
  // Create a no-op setCenter function for external use
  // Map center is now fixed at Denver and cannot be changed externally
  const nopSetCenter = useCallback((newCenter: Point) => {
    console.log('setCenter called but ignored - map stays at Denver:', newCenter);
    // Do nothing - map center stays fixed at Denver
  }, []);
  const [tifUrl, setTifUrl] = useState<string | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [isLoaded, setLoaded] = useBoolean();
  const [overlayData, setOverlayData] = useState<OverlayData | null>(null);
  const [closureMode, setClosureMode] = useBoolean();
  const { createClosure, closureData, createLoading, delClosure } = useClosureMarkers();
  // 调试日志：检查closure数据
  useEffect(() => {
    console.log('Closure data loaded:', closureData);
    console.log('Closure data length:', closureData?.length || 0);
    console.log('isLoaded:', isLoaded);
    if (closureData && closureData.length > 0) {
      console.log('First closure item:', closureData[0]);
    }
  }, [closureData, isLoaded]);
  const [currentClosure, setCurrentClosure] = useState<IRoadClosure | null>(null);
  const [route, setRouteState] = useState<IRoutePath | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentCustomFire, setCurrentCustomFire] = useState<CustomFirePoint | null>(null);
  const [currentShelter, setCurrentShelter] = useState<ShelterData | null>(null);
  const [showShelters, setShowShelters] = useState<boolean>(true); // Changed to true for testing
  const [shelterRefreshKey, setShelterRefreshKey] = useState(0);
  const [shelters, setShelters] = useState<ShelterData[]>([]);
  const [sheltersLoading, setSheltersLoading] = useState(false);
  
  // Map theme and type states
  const [mapTheme, setMapTheme] = useState<MapTheme>('original'); // Default to original theme
  const [mapType, setMapType] = useState<'roadmap' | 'satellite' | 'hybrid' | 'terrain'>('roadmap');

  // Simplified shelter data fetching - directly from database
  const fetchShelters = useCallback(async () => {
    setSheltersLoading(true);

    try {
      const response = await fetch('/api/shelters');
      const result = await response.json();

      // Always set the data, even if it's empty
      const shelterData = result.data || [];
      setShelters(shelterData);

    } catch (error) {
      console.error('Error fetching shelters:', error);
      setShelters([]);
    } finally {
      setSheltersLoading(false);
    }
  }, []);

  // Fetch shelters on mount and when refresh key changes
  useEffect(() => {
    fetchShelters();
  }, [fetchShelters, shelterRefreshKey]);



  // Force refresh shelters when showShelters becomes true
  useEffect(() => {
    if (showShelters) {
      console.log('showShelters became true, refreshing shelter data...');
      fetchShelters();
    }
  }, [showShelters, fetchShelters]);

  // Enhanced refresh function that also forces re-render
  const enhancedRefreshShelters = useCallback(async () => {
    await fetchShelters();
  }, [fetchShelters]);
  // 添加一个强制清除标志
  const [forceClear, setForceClear] = useState(false);
  // 自定义setRoute函数，处理强制清除
  const setRoute = useCallback((newRoute: IRoutePath | null) => {
    if (newRoute === null) {
      // 清除路径时，先设置强制清除标志
      setForceClear(true);
      setTimeout(() => {
        setRouteState(null);
        setForceClear(false);
        // 清除路径后，回到丹佛中心
        if (map) {
          console.log("🔄 清除路径，回到丹佛中心，缩放级别5.5");
          map.setCenter(DEFAULT_CENTER);
          map.setZoom(5.4);
        }
      }, 100);
    } else {
      setRouteState(newRoute);
    }
  }, [map]);

  const onLoad = useCallback((map: google.maps.Map) => {
    // This is just an example of getting and using the map instance!!! don't just blindly copy!
    // const bounds = new window.google.maps.LatLngBounds(center);
    // map.fitBounds(bounds);
    setMap(map);
    setLoaded.setTrue();
    
    // Force map resize after a short delay to ensure proper rendering
    setTimeout(() => {
      if (map && window.google?.maps?.event) {
        window.google.maps.event.trigger(map, 'resize');
        map.setCenter(DEFAULT_CENTER); // Always set to Denver
      }
    }, 100);
  }, []); // Remove center dependency

  const canvasSize = useMemo(() => {
    if (overlayData) {
      const { width, height } = overlayData;

      return { width, height };
    }

    return { width: 0, height: 0 };
  }, [overlayData]);

  const Info = useMemo(() => {
    if (!currentMarker) {
      return null;
    }

    const { raw, lng, lat } = currentMarker!;
    
    // Format the fire data for display
    const formatFireData = (raw: any) => {
      const data = {
        basic: [
          { label: "Date", value: raw.date || "Unknown" },
          { label: "Location", value: `${lat.toFixed(4)}, ${lng.toFixed(4)}` },
          { label: "Acres Burned", value: raw.acres ? `${raw.acres} acres` : "Unknown" },
          { label: "Containment", value: raw.containper ? `${raw.containper}%` : "0%" },
        ],
        detection: [
          { label: "MODIS", value: raw.modis === "yes" ? "✓" : "✗" },
          { label: "VIIRS", value: raw.viirs === "yes" ? "✓" : "✗" },
          { label: "WFIGS", value: raw.wfigs === "yes" ? "✓" : "✗" },
          { label: "FIRIS", value: raw.firis === "yes" ? "✓" : "✗" },
          { label: "Fire Guard", value: raw["fireguard "] === "yes" ? "✓" : "✗" },
        ]
      };
      return data;
    };

    const fireData = formatFireData(raw);
    const fireName = raw.prettyname || raw.name || "Unknown Fire";

    return (
      <InfoWindow
        options={{ 
          pixelOffset: new window.google.maps.Size(0, -30),
          disableAutoPan: false,
          headerDisabled: true
        }}
        position={{ lat, lng }}
      >
        <div className="eva-card" style={{ 
          width: '350px', 
          margin: 0, 
          padding: 'var(--spacing-md)',
          position: 'relative'
        }}>
          {/* Close button */}
          <button 
            onClick={() => setCurrentMarker(null)}
            style={{
              position: 'absolute',
              top: 'var(--spacing-sm)',
              right: 'var(--spacing-sm)',
              background: 'none',
              border: 'none',
              fontSize: '18px',
              color: 'var(--color-text-secondary)',
              cursor: 'pointer',
              padding: 'var(--spacing-xs)',
              borderRadius: '50%',
              width: '28px',
              height: '28px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            title="Close"
          >
            ×
          </button>

          {/* Header */}
          <div className="eva-flex" style={{ gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)', paddingRight: '30px' }}>
            <h3 className="eva-heading" style={{ fontSize: '16px', margin: 0, color: '#dc2626' }}>
              {fireName}
            </h3>
          </div>

          {/* Basic Information */}
          <div style={{ marginBottom: 'var(--spacing-md)' }}>
            <h4 style={{ 
              fontSize: '14px', 
              color: 'var(--color-text-secondary)', 
              margin: '0 0 var(--spacing-sm) 0',
              fontWeight: '600'
            }}>
              Basic Information
            </h4>
            <div style={{ 
              backgroundColor: 'var(--color-background-light)',
              padding: 'var(--spacing-sm)',
              borderRadius: 'var(--border-radius)',
              fontSize: '13px'
            }}>
              {fireData.basic.map((item, index) => (
                <div key={index} className="eva-flex eva-flex--between" style={{ 
                  padding: '4px 0',
                  borderBottom: index < fireData.basic.length - 1 ? '1px solid var(--color-border-light)' : 'none'
                }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>{item.label}:</span>
                  <span style={{ fontWeight: '500' }}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Detection Systems */}
          <div style={{ marginBottom: 'var(--spacing-md)' }}>
            <h4 style={{ 
              fontSize: '14px', 
              color: 'var(--color-text-secondary)', 
              margin: '0 0 var(--spacing-sm) 0',
              fontWeight: '600'
            }}>
              Detection Systems
            </h4>
            <div style={{ 
              backgroundColor: 'var(--color-background-light)',
              padding: 'var(--spacing-sm)',
              borderRadius: 'var(--border-radius)',
              fontSize: '13px'
            }}>
              {fireData.detection.map((item, index) => (
                <div key={index} className="eva-flex eva-flex--between" style={{ 
                  padding: '4px 0',
                  borderBottom: index < fireData.detection.length - 1 ? '1px solid var(--color-border-light)' : 'none'
                }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>{item.label}:</span>
                  <span style={{ 
                    fontWeight: '500',
                    color: item.value === "✓" ? '#10b981' : '#ef4444'
                  }}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </InfoWindow>
    );
  }, [currentMarker]);

  // COMMENTED OUT: Fire Perimeter info cards - keeping code but disabling functionality
  // const CustomFireInfo = useMemo(() => {
  //   if (!currentCustomFire) {
  //     return null;
  //   }

  //   const { lat, lng, properties } = currentCustomFire;
  //   const data: { key: string; value: string }[] = [];

  //   data.push({ key: "Object ID", value: properties.OBJECTID.toString() });
  //   data.push({ key: "Type", value: properties.type });
  //   data.push({ key: "Area", value: `${properties.Shape__Area.toFixed(2)} sq units` });
  //   data.push({ key: "Perimeter", value: `${properties.Shape__Length.toFixed(2)} units` });
  //   data.push({ key: "Location", value: `${lat.toFixed(6)}, ${lng.toFixed(6)}` });

  //   return (
  //     <InfoWindow
  //       options={{ pixelOffset: new window.google.maps.Size(0, -30) }}
  //       position={{ lat, lng }}
  //       onCloseClick={() => {
  //         setCurrentCustomFire(null);
  //       }}
  //     >
  //       <div className="w-[300px] bg-white bg-opacity-85 backdrop-blur-md rounded-lg p-4 shadow-xl border border-white border-opacity-30">
  //         <h3 className="text-lg font-bold mb-3 text-red-600 drop-shadow-sm">🔥 Fire Perimeter</h3>
  //         <div className="bg-white bg-opacity-40 rounded-md p-2">
  //           <Table
  //             bordered
  //             className="transparent-table"
  //             columns={[
  //               { title: "Property", dataIndex: "key", key: "key", width: "40%" },
  //               { title: "Value", dataIndex: "value", key: "value", width: "60%" },
  //             ]}
  //             dataSource={data}
  //             pagination={false}
  //             size="small"
  //           />
  //         </div>
  //       </div>
  //     </InfoWindow>
  //   );
  // }, [currentCustomFire]);
  
  // Fire Perimeter info disabled - return null to hide info cards
  const CustomFireInfo = null;

  const ShelterInfo = useMemo(() => {
    if (!currentShelter) {
      return null;
    }

    const { lat, lng, shelterId, capacity, region } = currentShelter;
    
    // Format the shelter data for display
    const shelterData = {
      basic: [
        { label: "Shelter ID", value: shelterId },
        { label: "Region", value: region.toUpperCase() },
        { label: "Capacity", value: `${capacity} people` },
        { label: "Location", value: `${lat.toFixed(6)}, ${lng.toFixed(6)}` },
      ]
    };

    return (
      <InfoWindow
        options={{ 
          pixelOffset: new window.google.maps.Size(0, -30),
          disableAutoPan: false,
          headerDisabled: true
        }}
        position={{ lat, lng }}
      >
        <div className="eva-card" style={{ 
          width: '350px', 
          margin: 0, 
          padding: 'var(--spacing-md)',
          position: 'relative'
        }}>
          {/* Close button */}
          <button 
            onClick={() => setCurrentShelter(null)}
            style={{
              position: 'absolute',
              top: 'var(--spacing-sm)',
              right: 'var(--spacing-sm)',
              background: 'none',
              border: 'none',
              fontSize: '18px',
              color: 'var(--color-text-secondary)',
              cursor: 'pointer',
              padding: 'var(--spacing-xs)',
              borderRadius: '50%',
              width: '28px',
              height: '28px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            title="Close"
          >
            ×
          </button>

          {/* Header */}
          <div className="eva-flex" style={{ gap: 'var(--spacing-sm)', marginBottom: 'var(--spacing-md)', paddingRight: '30px' }}>
            <h3 className="eva-heading" style={{ fontSize: '16px', margin: 0, color: '#2563eb' }}>
              Emergency Shelter
            </h3>
          </div>

          {/* Basic Information */}
          <div style={{ marginBottom: 'var(--spacing-md)' }}>
            <h4 style={{ 
              fontSize: '14px', 
              color: 'var(--color-text-secondary)', 
              margin: '0 0 var(--spacing-sm) 0',
              fontWeight: '600'
            }}>
              Shelter Information
            </h4>
            <div style={{ 
              backgroundColor: 'var(--color-background-light)',
              padding: 'var(--spacing-sm)',
              borderRadius: 'var(--border-radius)',
              fontSize: '13px'
            }}>
              {shelterData.basic.map((item, index) => (
                <div key={index} className="eva-flex eva-flex--between" style={{ 
                  padding: '4px 0',
                  borderBottom: index < shelterData.basic.length - 1 ? '1px solid var(--color-border-light)' : 'none'
                }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>{item.label}:</span>
                  <span style={{ fontWeight: '500' }}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </InfoWindow>
    );
  }, [currentShelter]);

  const ClosureInfo = useMemo(() => {
    if (!currentClosure) {
      return null;
    }

    const { lng, lat } = currentClosure!;
    const data: { key: string; value: string }[] = [];

    data.push({ key: "Location", value: `${lat}, ${lng}` });
    data.push({ key: "Created At", value: dayjs(currentClosure.createdAt).format("YYYY-MM-DD HH:mm:ss") });
    data.push({ key: "User", value: currentClosure.User.username });

    return (
      <InfoWindow
        options={{ pixelOffset: new window.google.maps.Size(0, -30) }}
        position={{ lat, lng }}
        onCloseClick={() => {
          setCurrentClosure(null);
        }}
      >
        <div className="flex flex-col gap-y-5">
          <Table
            bordered
            className="w-[400px]"
            columns={[
              { title: "", dataIndex: "key", key: "key", colSpan: 0, className: "p-2! h-10!" },
              { title: "", dataIndex: "value", key: "value", colSpan: 0, className: "p-2! h-10!" },
            ]}
            dataSource={data}
            pagination={false}
          />
          <div className="flex items-center justify-end">
            <Popconfirm
              cancelText="No"
              description="Are you sure to delete this?"
              okText="Yes"
              title="Delete"
              // onCancel={cancel}
              onConfirm={() => {
                delClosure(currentClosure.id).then(() => {
                  setCurrentClosure(null);
                });
              }}
            >
              <Button danger>Delete</Button>
            </Popconfirm>
          </div>
        </div>
      </InfoWindow>
    );
  }, [currentClosure]);

  const overlay = useMemo(() => {
    if (overlayData) {
      const bounds = overlayData.bbox;

      return (
        <GroundOverlay
          bounds={bounds}
          url={overlayData.data}
          onClick={(e) => {
            const lat = e.latLng!.lat();
            const lng = e.latLng!.lng();
            const img = document.createElement("img");

            img.src = overlayData.data;
            img.onload = () => {
              const canvas = canvasRef.current!;
              const { width, height } = img;

              canvas.width = width;
              canvas.height = height;

              const ctx = canvas.getContext("2d")!;

              ctx.clearRect(0, 0, width, height);

              ctx.drawImage(img, 0, 0);
              const x = Math.floor(((lng - bounds.west) / (bounds.east - bounds.west)) * canvas.width);
              const y = Math.floor(((bounds.north - lat) / (bounds.north - bounds.south)) * canvas.height);

              const pixel = ctx.getImageData(x, y, 1, 1).data;
              const [r, g, b, a] = pixel;

              if (r != 0 || g != 0 || b != 0 || a != 0) {
                // 计算原始值（反向映射）
                const minValue = overlayData.min; // 来自 jsonData
                const maxValue = overlayData.max; // 来自 jsonData
                const t = r / 255;
                const originalValue = maxValue - t * (maxValue - minValue);

                if (!overlayWindow) {
                  overlayWindow = new google.maps.InfoWindow();
                }
                const unit = getUnit(tifUrl!);

                overlayWindow.setPosition({ lat, lng });
                overlayWindow.setContent(`${originalValue.toFixed(2)} ${unit}`);
                overlayWindow.open({
                  map,
                });
                // console.log(`估算原始值（火点强度）：${originalValue.toFixed(2)}`);
              }
            };
          }}
        />
      );
    }

    return null;
  }, [overlayData, map, tifUrl]);
  // 路径版本计数器，用于强制重新渲染
  const [routeVersion, setRouteVersion] = useState(0);
  // 存储当前的polyline实例
  const [polylineInstance, setPolylineInstance] = useState<google.maps.Polyline | null>(null);

  const routeComponents = useMemo(() => {
    // 如果强制清除或没有路径，返回null确保组件被完全移除
    if (forceClear || !route) {
      return null;
    }

    const path = route.route.map(([lat, lng]) => ({ lat, lng }));
    const startPoint = { lat: route.start[0], lng: route.start[1] };

    return (
      <>
        {/* 路径线 */}
        <Polyline
          key={`route-${routeVersion}`}
          options={{
            strokeColor: "#FF0000",
            strokeWeight: 3,
            strokeOpacity: 0.9,
            zIndex: 1000,
          }}
          path={path}
          onLoad={(polyline) => {
            setPolylineInstance(polyline);
          }}
          onUnmount={() => {
            setPolylineInstance(null);
          }}
        />
        {/* 起点用户标记 */}
        <Marker
          key={`start-point-${routeVersion}`}
          icon={{ url: "/user.png", scaledSize: new window.google.maps.Size(200, 200) }}
          position={startPoint}
          title="Start Point"
          zIndex={1001}
        />
      </>
    );
  }, [route, routeVersion, forceClear]);

  // 自定义火点多边形
  const customFirePolygon = useMemo(() => {
    if (!currentCustomFire) {
      return null;
    }

    const coordinates = currentCustomFire.geometry.coordinates[0];
    const path = coordinates.map(([lng, lat]) => ({ lat, lng }));

    return (
      <Polygon
        key={`custom-fire-${currentCustomFire.id}`}
        options={{
          fillColor: "#FF6600",
          fillOpacity: 0.3,
          strokeColor: "#FF6600",
          strokeWeight: 2,
          strokeOpacity: 0.8,
        }}
        paths={path}
      />
    );
  }, [currentCustomFire]);

  // 当route改变时，更新版本号
  useEffect(() => {
    setRouteVersion((prev) => prev + 1);
  }, [route]);

  // 当强制清除时，直接操作Google Maps API
  useEffect(() => {
    if (forceClear && polylineInstance) {
      polylineInstance.setMap(null);
      setPolylineInstance(null);
    }
  }, [forceClear, polylineInstance]);
  const polyline = useMemo(() => {
    if (!route) {
      return null;
    }

    const path = route.route.map(([lat, lng]) => ({ lat, lng }));

    return (
      <>
        {/* <Marker
          icon={{ url: "/user.png", scaledSize: new window.google.maps.Size(42, 42) }}
          position={{ lat: route.start[0], lng: route.start[1] }}
        /> */}
        <Polyline
          options={{
            strokeColor: "#FF0000",
            // fillOpacity: 0.5,
            strokeWeight: 2,
          }}
          path={path}
        />
      </>
    );
  }, [route]);

  // 适应覆盖区域
  useEffect(() => {
    if (map && overlayData) {
      // 4. 调整视角到覆盖区域
      const bounds = new google.maps.LatLngBounds(
        new google.maps.LatLng(overlayData.bbox.south, overlayData.bbox.west),
        new google.maps.LatLng(overlayData.bbox.north, overlayData.bbox.east),
      );

      map.fitBounds(bounds);
    }
  }, [overlayData, map]);

  //  fit逃生路线 - 优化版本 - 路径规划时允许地图缩放
  useEffect(() => {
    if (route && map) {
      console.log("🔍 开始自动缩放到路径位置...", route);

      const path = route.route.map(([lat, lng]) => ({ lat, lng }));

      // 创建边界框包含整个路径
      const bounds = new google.maps.LatLngBounds();

      path.forEach(({ lat, lng }) => {
        bounds.extend(new google.maps.LatLng(lat, lng));
      });

      // 添加起点和终点确保完整覆盖
      bounds.extend(new google.maps.LatLng(route.start[0], route.start[1]));
      bounds.extend(new google.maps.LatLng(route.destination[0], route.destination[1]));

      console.log("📍 路径边界:", {
        start: route.start,
        destination: route.destination,
        pathPoints: path.length,
      });

      // 使用动画效果平滑缩放到路径 - 添加延迟确保路径已渲染
      setTimeout(() => {
        map.fitBounds(bounds, {
          top: 80, // 顶部边距
          right: 80, // 右侧边距
          bottom: 80, // 底部边距
          left: 80, // 左侧边距
        });

        // 延迟设置最小缩放级别，避免过度放大
        setTimeout(() => {
          const currentZoom = map.getZoom();
          console.log("🔍 当前缩放级别:", currentZoom);
          if (currentZoom && currentZoom > 15) {
            console.log("⚠️ 缩放级别过高，调整到15级");
            map.setZoom(15);
          }
        }, 300);
      }, 200);
    }
  }, [route, map]);

  // fit到Marker - 禁用自动缩放到火点位置，保持地图在丹佛中心
  // useEffect(() => {
  //   if (markers.length > 0 && map && !route) {
  //     console.log("🔍 自动缩放到火点标记位置...");
  //     const bounds = new google.maps.LatLngBounds();

  //     markers.forEach(({ lat, lng }) => {
  //       bounds.extend(new google.maps.LatLng(lat, lng));
  //     });
  //     map.fitBounds(bounds);
  //   }
  // }, [markers, map, route]);

  // Handle window resize and map refresh
  useEffect(() => {
    const handleResize = () => {
      if (map && window.google?.maps?.event) {
        setTimeout(() => {
          window.google.maps.event.trigger(map, 'resize');
          map.setCenter(DEFAULT_CENTER); // Always center at Denver
        }, 100);
      }
    };

    window.addEventListener('resize', handleResize);
    
    // Also trigger resize after component mount
    const timer = setTimeout(handleResize, 200);

    return () => {
      window.removeEventListener('resize', handleResize);
      clearTimeout(timer);
    };
  }, [map]); // Remove center dependency

  // Map theme styles
  const mapThemeStyles = {
    'darkview': [
      {
        featureType: "all",
        elementType: "labels.text.fill",
        stylers: [{ saturation: 36 }, { color: "#000000" }, { lightness: 40 }]
      },
      {
        featureType: "all",
        elementType: "labels.text.stroke",
        stylers: [{ visibility: "on" }, { color: "#000000" }, { lightness: 16 }]
      },
      {
        featureType: "all",
        elementType: "labels.icon",
        stylers: [{ visibility: "off" }]
      },
      {
        featureType: "administrative",
        elementType: "geometry.fill",
        stylers: [{ color: "#000000" }, { lightness: 20 }]
      },
      {
        featureType: "administrative",
        elementType: "geometry.stroke",
        stylers: [{ color: "#000000" }, { lightness: 17 }, { weight: 1.2 }]
      },
      {
        featureType: "landscape",
        elementType: "geometry",
        stylers: [{ color: "#000000" }, { lightness: 20 }]
      },
      {
        featureType: "poi",
        elementType: "geometry",
        stylers: [{ color: "#000000" }, { lightness: 21 }]
      },
      {
        featureType: "road.highway",
        elementType: "geometry.fill",
        stylers: [{ color: "#ff4500" }, { lightness: 17 }]
      },
      {
        featureType: "road.highway",
        elementType: "geometry.stroke",
        stylers: [{ color: "#000000" }, { lightness: 29 }, { weight: 0.2 }]
      },
      {
        featureType: "road.arterial",
        elementType: "geometry",
        stylers: [{ color: "#000000" }, { lightness: 18 }]
      },
      {
        featureType: "road.local",
        elementType: "geometry",
        stylers: [{ color: "#000000" }, { lightness: 16 }]
      },
      {
        featureType: "transit",
        elementType: "geometry",
        stylers: [{ color: "#000000" }, { lightness: 19 }]
      },
      {
        featureType: "water",
        elementType: "geometry",
        stylers: [{ color: "#1a4f7a" }, { lightness: 17 }]
      }
    ],
    'original': [] // Empty array means no custom styles, showing Google's default map
  };

  // Apply the selected map theme styles
  useEffect(() => {
    if (map && mapTheme) {
      const styles = mapThemeStyles[mapTheme];

      if (styles) {
        map.setOptions({ styles });
      }
    }
  }, [map, mapTheme, mapThemeStyles]);

  // Apply map theme styles when theme or map changes
  useEffect(() => {
    if (map && mapTheme) {
      const styles = mapThemeStyles[mapTheme];
      map.setOptions({ 
        styles: styles,
        mapTypeId: mapType
      });
    }
  }, [map, mapTheme, mapType]);

  // Apply dark theme to sidebar when darkview map is active
  useEffect(() => {
    const isDarkTheme = mapTheme === 'darkview';
    const root = document.documentElement;
    
    if (isDarkTheme) {
      // Apply dark theme CSS variables
      root.style.setProperty('--color-background', '#1E201E');
      root.style.setProperty('--color-background-gray', '#2d2d2d');
      root.style.setProperty('--color-text-primary', '#E9EED9');
      root.style.setProperty('--color-text-secondary', '#E9EED9');
      root.style.setProperty('--color-text-muted', '#8a8a8a');
      root.style.setProperty('--color-border', '#404040');
      root.style.setProperty('--color-border-light', '#333333');
      root.style.setProperty('--color-card-bg', '#2d2d2d');
      root.style.setProperty('--color-primary', '#aec8a4');
      root.style.setProperty('--color-primary-light', '#31511e');
      root.style.setProperty('--color-primary-hover', '#6B7F2D');
    } else {
      // Reset to light theme CSS variables
      root.style.setProperty('--color-background', '#fffbf8');
      root.style.setProperty('--color-background-gray', '#8eb486');
      root.style.setProperty('--color-text-primary', '#3b3030');
      root.style.setProperty('--color-text-secondary', '#685752');
      root.style.setProperty('--color-text-muted', '#997c70');
      root.style.setProperty('--color-border', '#ab886d');
      root.style.setProperty('--color-border-light', '#d6c0b3');
      root.style.setProperty('--color-card-bg', '#fffbf8');
      root.style.setProperty('--color-primary', '#31511e');
      root.style.setProperty('--color-primary-light', '#aec8a4');
      root.style.setProperty('--color-primary-hover', '#8eb486');
    }
  }, [mapTheme]);

  return (
    <MapContext.Provider
      value={{
        setTifUrl,
        loaded: isLoaded,
        setCenter: nopSetCenter, // Use no-op function instead of real setCenter
        currentMarker,
        setCurrentMarker,
        currentCustomFire,
        setCurrentCustomFire,
        currentShelter,
        setCurrentShelter,
        showShelters,
        setShowShelters,
        refreshShelters: enhancedRefreshShelters,
        map: map!,
        setOverlayData,
        closureMode,
        setClosureMode,
        route,
        setRoute,
        // Map theme and type properties
        mapTheme,
        mapType,
        setMapTheme: (theme: MapTheme) => {
          setMapTheme(theme);
          // Immediately apply the theme
          if (map) {
            const styles = mapThemeStyles[theme];
            map.setOptions({ styles });
          }
        },
        setMapType: (type: 'roadmap' | 'satellite' | 'hybrid' | 'terrain') => {
          setMapType(type);
          if (map) {
            map.setMapTypeId(type);
          }
        },
        // User location control
        useUserLocation,
        setUseUserLocation,
        resetToDefaultCenter,
        
        // Simulated user location for address switching
        simulatedUserLocation,
        setSimulatedUserLocation,
      }}
    >
      <LoadScript googleMapsApiKey={process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY!}
      language="en"        // 设置界面语言为英文
      region="US"  >
        <GoogleMap
          center={DEFAULT_CENTER} // Fixed center at Denver
          mapContainerStyle={containerStyle}
          zoom={5.4}
          options={{
            styles: [], // Default to no custom styles (original map)
            mapTypeId: 'roadmap',
            mapTypeControl: false, // Hide the default map/satellite toggle
            streetViewControl: false, // Also hide street view control for cleaner UI
            fullscreenControl: true, // Keep fullscreen control
            zoomControl: true, // Keep zoom controls
          }}
          onClick={(e) => {
            if (closureMode && !createLoading) {
              // 设定封路
              createClosure(e.latLng!.toJSON());
            }
            setCurrentMarker(null);
          }}
          onLoad={onLoad}
        >
          {isLoaded &&
            markers.map((m, index) => {
              // Debug log to see what markers we're getting
              if (index < 5) {
                console.log('Rendering marker:', m);
              }
              return (
                <Marker
                  key={index}
                  icon={{ url: "/fire.png", scaledSize: new window.google.maps.Size(32, 32) }}
                  position={{ lat: m.lat, lng: m.lng }}
                  onClick={() => {
                    setCurrentMarker(m);
                    setCurrentClosure(null);
                  }}
                />
              );
            })}
          {isLoaded &&
            reportMarkers.map((m, index) => {
              return (
                <Marker
                  key={index}
                  icon={{ url: "/fire.png", scaledSize: new window.google.maps.Size(32, 32) }}
                  position={{ lat: m.lat, lng: m.lng }}
                  onClick={(e) => {
                    const container = document.createElement("div");

                    // 使用 React 18 的 root API
                    const root = createRoot(container);

                    root.render(
                      <Button
                        type="primary"
                        onClick={() => {
                          delReport(m.id).then(() => {
                            // 处理删除成功后的逻辑
                            info.close();
                          });
                        }}
                      >
                        删除
                      </Button>,
                    );

                    var info = new google.maps.InfoWindow({
                      pixelOffset: new window.google.maps.Size(0, -30),
                      position: { lat: m.lat, lng: m.lng },
                      content: container,
                    });

                    // new window.google.maps.Size(0, -30)
                    // 将 MyComponent 转换成 HTML 字符串
                    info.open(map);
                  }}
                />
              );
            })}
          {isLoaded &&
            closureData.map((m, index) => {
              console.log(`Rendering closure marker ${index}:`, m);
              return (
                <Marker
                  key={`closure-${index}-${m.id || index}`}
                  icon={{ url: "/barrier.png", scaledSize: new window.google.maps.Size(42, 42) }}
                  position={{ lat: m.lat, lng: m.lng }}
                  title={`Road Closure ${m.id || index} - Click for details`}
                  onClick={() => {
                    console.log('Closure marker clicked:', m);
                    setCurrentMarker(null);
                    setCurrentClosure(m);
                  }}
                />
              );
            })}
          {/* Shelter markers - with pulse animation circles */}
          {isLoaded && showShelters && shelters.map((shelter, index) => {
            const lat = parseFloat(shelter.lat.toString());
            const lng = parseFloat(shelter.lng.toString());
            
            return (
              <Fragment key={`shelter-${shelter.id}`}>
                {/* Animated pulse circles */}
                <Marker
                  icon={{
                    url: `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`
                      <svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="60" cy="60" r="50" fill="#859F3D" fill-opacity="0.1" stroke="#859F3D" stroke-width="2" stroke-opacity="0.2">
                          <animate attributeName="r" values="30;50;30" dur="2s" repeatCount="indefinite"/>
                          <animate attributeName="fill-opacity" values="0.3;0.05;0.3" dur="2s" repeatCount="indefinite"/>
                          <animate attributeName="stroke-opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite"/>
                        </circle>
                        <circle cx="60" cy="60" r="35" fill="#859F3D" fill-opacity="0.15" stroke="#859F3D" stroke-width="1" stroke-opacity="0.3">
                          <animate attributeName="r" values="20;35;20" dur="2s" begin="0.5s" repeatCount="indefinite"/>
                          <animate attributeName="fill-opacity" values="0.25;0.08;0.25" dur="2s" begin="0.5s" repeatCount="indefinite"/>
                        </circle>
                      </svg>
                    `)}`,
                    scaledSize: new window.google.maps.Size(120, 120),
                    anchor: new window.google.maps.Point(60, 60)
                  }}
                  position={{ lat, lng }}
                  zIndex={1}
                />
                {/* Main shelter icon */}
                <Marker
                  icon={{
                    url: "/shelter2.png",
                    scaledSize: new window.google.maps.Size(32, 32),
                    anchor: new window.google.maps.Point(16, 16)
                  }}
                  position={{ lat, lng }}
                  title={`Shelter ${shelter.id} - ${shelter.region}`}
                  zIndex={10}
                  onClick={() => {
                    setCurrentMarker(null);
                    setCurrentClosure(null);
                    setCurrentShelter(shelter);
                  }}
                />
              </Fragment>
            );
          })}
          {/* Custom fire markers - COMMENTED OUT: click events disabled */}
          {isLoaded && customFires.map((fire, index) => (
            <Marker
              key={`custom-fire-${fire.id}`}
              icon={{
                url: "/fire.png",
                scaledSize: new window.google.maps.Size(32, 32)
              }}
              position={{
                lat: fire.lat,
                lng: fire.lng
              }}
              title={`Custom Fire ${fire.id} - ${fire.properties.type}`}
              // COMMENTED OUT: onClick event disabled to prevent selection
              // onClick={() => {
              //   setCurrentMarker(null);
              //   setCurrentClosure(null);
              //   setCurrentShelter(null);
              //   setCurrentCustomFire(fire);
              // }}
            />
          ))}
          {/* Simulated user location marker for address switching */}
          {isLoaded && simulatedUserLocation && (
            <Marker
              key={`simulated-user-location-${simulatedUserLocation.lat}-${simulatedUserLocation.lng}`}
              icon={{
                url: "/user.png",
                scaledSize: new window.google.maps.Size(50, 50)
              }}
              position={{
                lat: simulatedUserLocation.lat,
                lng: simulatedUserLocation.lng
              }}
              title={`Selected Location: ${simulatedUserLocation.address} (${simulatedUserLocation.lat.toFixed(6)}, ${simulatedUserLocation.lng.toFixed(6)})`}
              onClick={() => {
                setCurrentMarker(null);
                setCurrentClosure(null);
                setCurrentShelter(null);
                setCurrentCustomFire(null);
              }}
            />
          )}
          {/* Current user location marker - only show if different from simulated location */}
          {/* {isLoaded && currentLocation && isValidCoordinate(currentLocation.lat, currentLocation.lng) && 
           (!simulatedUserLocation || 
            (currentLocation.lat !== simulatedUserLocation.lat || currentLocation.lng !== simulatedUserLocation.lng)) && (
            <Marker
              key={`user-location-${userLocationKey}-${currentLocation.lat}-${currentLocation.lng}-${Date.now()}`}
              icon={{
                url: "/user0.png", // Use different icon to distinguish from simulated location
                scaledSize: new window.google.maps.Size(40, 40)
              }}
              position={{
                lat: currentLocation.lat as number,
                lng: currentLocation.lng as number
              }}
              title={`Your Actual Location (${currentLocation.lat}, ${currentLocation.lng}) - ${currentLocation.username}`}
              onClick={() => {
                setCurrentMarker(null);
                setCurrentClosure(null);
                setCurrentShelter(null);
                setCurrentCustomFire(null);
              }}
            />
          )} */}
          {Info}
          {ShelterInfo}
          {ClosureInfo}
          {CustomFireInfo}
          {customFirePolygon}
          {overlay}
          {polyline}
        </GoogleMap>
      </LoadScript>
      
      {/* Map Controls */}
      <MapControls />
      
      {children}


      <canvas
        ref={canvasRef}
        className="absolute -z-10 left-0 top-0"
        height={canvasSize.height}
        width={canvasSize.width}
      />
    </MapContext.Provider>
  );
};

export const useMapContext = () => {
  const context = useContext(MapContext);

  if (!context) {
    throw new Error("useMapContext must be used within a MapProvider");
  }

  return context;
};
