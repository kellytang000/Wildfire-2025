"use client";

import { useBoolean } from "ahooks";
import { Button, Card, Form, Input, Modal, Cascader, Spin } from "antd";
import { FC, PropsWithChildren, useEffect, useMemo, useState } from "react";

import ChatWindow from "../ai/ChatWindow";
// import { UserLocationPanel } from "../UserLocationPanelModern"; // Commented out - location info now in LocationInfoCard
import LocationInfoCard from "../cards/LocationInfoCard";
import StatsCard from "../cards/StatsCard";
import MobileMenuToggle from "../MobileMenuToggle";

import { useMapContext } from "@/src/context/map.ctx";
import { useUser } from "@/src/hooks/useUser";
import { useEscape } from "@/src/actions/closure.hook";
import ShelterManager from "@/src/components/ShelterManager";
import { useRouter } from "next/navigation";

// Import for testing Fire Spread Forecast in sidebar
import { Option, useOptions } from "@/src/actions/tif.hook";
import { useRequest } from "ahooks";
import { API } from "@/src/utils/http";

// Add OverlayData type definition
export type OverlayData = {
  data: string;
  bbox: {
    west: number;
    south: number;
    east: number;
    north: number;
  };
  width: number;
  height: number;
  max: number;
  min: number;
};

interface IProps {}
const ModernSidebar: FC<PropsWithChildren<IProps>> = () => {
  const router = useRouter();
  const {
    currentMarker,
    closureMode,
    setClosureMode,
    route,
    setRoute,
    showShelters,
    setShowShelters,
    refreshShelters,
    setCenter,
    setOverlayData,
    setTifUrl,
    useUserLocation,
    setUseUserLocation,
    resetToDefaultCenter,
  } = useMapContext();
  const { user, logout } = useUser();
  const [showShelterManager, setShowShelterManager] = useBoolean();

  // Testing Fire Spread Forecast in sidebar
  const { options, getOptions, loading: firstLoading } = useOptions();
  const { loading, runAsync: loadOption } = useOptions();
  const [values, setValues] = useState<(string | number | null)[]>([]);
  const [list, setList] = useState<Option[]>([]);

  const { run: loadTif, loading: loadingTif } = useRequest(
    (tifUrl: string, lat: number, lng: number) => {
      return API.get<OverlayData>("/tif", { data: { url: tifUrl, lat, lng } });
    },
    {
      manual: true,
      onSuccess(data) {
        setOverlayData(data);
      },
    },
  );

  const loadData = async (selectedOptions: Option[]) => {
    console.log('Sidebar loadData called with:', selectedOptions);
    const lastItem = selectedOptions[selectedOptions.length - 1];

    if (lastItem && currentMarker) {
      const target = selectedOptions.map((item) => item.value).join("");
      const url = `${currentMarker.raw.name}/${target}`;
      console.log('Sidebar loading data for URL:', url);

      try {
        const res = await loadOption(url);
        console.log('Sidebar loaded data:', res);
        
        if (res && res.length > 0) {
          lastItem.children = res;
        } else {
          // If no children found, mark as leaf node
          lastItem.isLeaf = true;
          lastItem.children = undefined;
        }
        
        // Create a new list to trigger re-render
        setList([...list]);
      } catch (error) {
        console.error('Sidebar error loading options:', error);
        // On error, mark as leaf node so it won't try to load again
        lastItem.isLeaf = true;
        lastItem.children = undefined;
        setList([...list]);
      }
    }
  };

  const onChange = async (vals: (string | number | null)[]) => {
    console.log('Sidebar onChange called with:', vals);
    setValues(vals);
    
    if (currentMarker && vals.length > 0) {
      const tifUrl = `${currentMarker.raw.name}/${vals.join("")}`;
      console.log('Sidebar generated TIF URL:', tifUrl);

      setTifUrl(tifUrl);
      setOverlayData(null);
      loadTif(tifUrl, currentMarker.lat, currentMarker.lng);
    }
  };

  useEffect(() => {
    if (currentMarker) {
      console.log('Sidebar current marker changed, initializing options for:', currentMarker.raw.name);
      setValues([]);
      getOptions(`${currentMarker.raw.name}/`, true);
    }
  }, [currentMarker, getOptions]);

  useEffect(() => {
    console.log('Sidebar options updated:', options);
    if (options.length) {
      setList(options);
    }
  }, [options]);

  const [isModalOpen, setIsModalOpen] = useBoolean();
  const { createEscape, loading: loadingEscape } = useEscape();
  const [chatVisible, setChatVisible] = useBoolean();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const [form] = Form.useForm();

  const handleClosure = async () => {
    const values = await form.validateFields();

    // 先清除旧路径
    setRoute(null);

    const res = await createEscape(values);

    // 添加小延迟确保状态更新
    setTimeout(() => {
      setRoute(res);
    }, 100);

    setRoute(res);
    setIsModalOpen.setFalse();
  };

  return (
    <>
      <MobileMenuToggle onToggle={setMobileMenuOpen} />
      <div className={`eva-sidebar ${mobileMenuOpen ? 'eva-sidebar--open' : ''}`}>
        <div className="eva-container eva-stack">
          {showShelterManager ? (
            <div className="fixed inset-0 bg-white z-50 overflow-auto">
              <ShelterManager />
              <Button className="fixed top-4 right-4 z-60" onClick={() => setShowShelterManager.setFalse()}>
                Close
              </Button>
            </div>
          ) : (
            <>
              {/* Location & Risk Info Card - Combined */}
              <LocationInfoCard 
                address={currentMarker ? `${currentMarker.lat.toFixed(4)}, ${currentMarker.lng.toFixed(4)}` : "Current Location"}
                region="Fire Emergency Zone"
                description="This area is under active wildfire monitoring. Evacuation routes are available and shelters are operational."
                coordinates={currentMarker ? { lat: currentMarker.lat, lng: currentMarker.lng } : undefined}
                currentPosition={currentMarker ? { lat: currentMarker.lat, lng: currentMarker.lng } : undefined}
                fireLevel="High"
                riskScore={7}
                riskDescription="Current weather conditions and dry vegetation increase wildfire risk."
              />

            {/* Fire Forecast Selection - Test Version in Sidebar */}
            <div className="eva-card">
              <div className="eva-flex eva-flex--between" style={{ marginBottom: 'var(--spacing-md)' }}>
                <h3 className="eva-heading">Fire Spread Forecast</h3>
                {loadingTif && <Spin size="small" />}
              </div>
            
                <div style={{ 
                  padding: 'var(--spacing-sm)', 
                  background: 'var(--color-background-light)', 
                  borderRadius: 'var(--border-radius)',
                  marginBottom: 'var(--spacing-md)',
                  fontSize: '13px',
                  color: 'var(--color-text-secondary)'
                }}>
                  💡 Please select a wildfire on the map first to view forecast models
                </div>
              
              <Cascader
                className="eva-input"
                loadData={loadData}
                loading={firstLoading || loading}
                options={list}
                placeholder={currentMarker ? "Select forecast model" : "Select a fire on map first"}
                value={values}
                onChange={onChange}
                expandTrigger="hover"
                changeOnSelect={false}
                disabled={!currentMarker}
                style={{ 
                  marginBottom: 'var(--spacing-sm)',
                  width: 'calc(100% - 0px)' // Full width with card padding
                }}
              />
            </div>

            {/* Combined Shelters Statistics and Controls */}
            <div className="eva-card">
              <div style={{ marginBottom: 'var(--spacing-md)' }}>
                <h3 className="eva-heading" style={{ marginBottom: 'var(--spacing-xs)' }}>Shelters Statistics</h3>
                
                {/* Header with Active Shelters and Hide/Show button */}
                <div className="eva-flex eva-flex--between eva-flex--center" style={{ marginBottom: 'var(--spacing-md)' }}>
                  <div style={{ 
                    background: 'var(--color-background-light)', 
                    padding: 'var(--spacing-sm)', 
                    borderRadius: 'var(--border-radius)',
                    flex: 1,
                    marginRight: 'var(--spacing-sm)'
                  }}>
                    <div>
                      <div className="eva-text eva-text--small eva-text--muted">Active Shelters</div>
                      <div style={{ fontSize: '24px', fontWeight: '600', color: 'var(--color-primary)' }}>
                        10 <span style={{ color: '#10b981', fontSize: '14px', marginLeft: '8px' }}>↗</span>
                      </div>
                    </div>
                  </div>
                  
                  <button 
                    className={`eva-button eva-button--small ${showShelters ? 'eva-button--primary' : 'eva-button--secondary'}`}
                    onClick={() => setShowShelters(!showShelters)}
                  >
                    {showShelters ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              {/* Controls */}
              {showShelters && (
                <div className="eva-flex" style={{ gap: 'var(--spacing-xs)' }}>
                  <button 
                    className="eva-button eva-button--ghost eva-button--small"
                    onClick={() => {
                      // Just show LA address info, don't change map center
                      console.log('LA address button clicked - display only');
                    }}
                    style={{ 
                      flex: 1,
                      border: '1px solid var(--color-border-light)',
                      backgroundColor: 'var(--color-background)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-primary)';
                      e.currentTarget.style.backgroundColor = 'var(--color-background-gray)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-border-light)';
                      e.currentTarget.style.backgroundColor = 'var(--color-background)';
                    }}
                  >
                    Check LA
                  </button>
                  <button 
                    className="eva-button eva-button--ghost eva-button--small"
                    onClick={() => {
                      // Just show BC address info, don't change map center  
                      console.log('BC address button clicked - display only');
                    }}
                    style={{ 
                      flex: 1,
                      border: '1px solid var(--color-border-light)',
                      backgroundColor: 'var(--color-background)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-primary)';
                      e.currentTarget.style.backgroundColor = 'var(--color-background-gray)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-border-light)';
                      e.currentTarget.style.backgroundColor = 'var(--color-background)';
                    }}
                  >
                    Check Squamish
                  </button>
                  <button 
                    className="eva-button eva-button--ghost eva-button--small"
                    onClick={() => refreshShelters()}
                    style={{ 
                      width: '40px', 
                      padding: '0',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#666',
                      border: '1px solid var(--color-border-light)',
                      backgroundColor: 'var(--color-background)',
                      transition: 'all 0.2s ease'
                    }}
                    title="Refresh Shelters"
                    onMouseOver={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-primary)';
                      e.currentTarget.style.backgroundColor = 'var(--color-background-gray)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-border-light)';
                      e.currentTarget.style.backgroundColor = 'var(--color-background)';
                    }}
                  >
                    ↻
                  </button>
                </div>
              )}
            </div>

            {/* Admin Controls */}
            {user && user.role === "ADMIN" && (
              <div className="eva-card">
                <h3 className="eva-heading" style={{ marginBottom: 'var(--spacing-md)' }}>Admin Controls</h3>
                <div className="eva-stack eva-stack--sm">
                  <button 
                    className={`eva-button ${closureMode ? 'eva-button--primary' : 'eva-button--secondary'}`}
                    onClick={() => setClosureMode.toggle()}
                  >
                    {closureMode ? "Done" : "Set Closure"}
                  </button>
                  <button 
                    className="eva-button eva-button--secondary"
                    onClick={() => setShowShelterManager.toggle()}
                  >
                    Manage Shelters
                  </button>
                  <button 
                    className="eva-button eva-button--secondary"
                    onClick={() => router.push('/admin/custom-fires')}
                  >
                    Manage Custom Fires
                  </button>
                </div>
              </div>
            )}

            {/* AI Chat & User Actions */}
                {user && (
                  <button 
                    className="eva-button eva-button--ghost"
                    onClick={logout}
                    style={{
                      border: '1px solid var(--color-primary)',
                      backgroundColor: 'var(--color-primary-light)',
                      color: 'var(--color-primary)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.backgroundColor = 'var(--color-primary)';
                      e.currentTarget.style.color = 'white';
                      e.currentTarget.style.borderColor = 'var(--color-primary)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.backgroundColor = 'var(--color-primary-light)';
                      e.currentTarget.style.color = 'var(--color-primary)';
                      e.currentTarget.style.borderColor = 'var(--color-primary)';
                    }}
                  >
                    Log Out
                  </button>
                )}
          </>
        )}
      </div>
      </div>

      {/* Floating AI Chat Window */}
      <div style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        zIndex: 99999,
        width: chatVisible ? '450px' : '80px',
        height: chatVisible ? '600px' : '80px',
        // transition: 'all 0.3s ease', // REMOVED: eliminates shrinking animation
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        borderRadius: chatVisible ? '12px' : '50%',
        overflow: 'hidden'
      }}>
        {chatVisible ? (
          <div style={{
            width: '100%',
            height: '100%',
            backgroundColor: 'white',
            display: 'flex',
            flexDirection: 'column'
          }}>
            {/* Chat Header */}
            <div style={{
              padding: '16px',
              borderBottom: '1px solid #f0f0f0',
              backgroundColor: '#fafafa',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  backgroundColor: '#F87A53',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '14px'
                }}>
                  <img 
                    src="/chatagent.png" 
                    alt="AI Assistant" 
                    style={{ 
                      width: '20px', 
                      height: '20px',
                      filter: 'brightness(0) invert(1)'
                    }} 
                  />
                </div>
                <span style={{ fontWeight: '500', fontSize: '14px' }}>AI Assistant</span>
              </div>
              <button
                onClick={setChatVisible.setFalse}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '16px',
                  color: '#666',
                  cursor: 'pointer',
                  padding: '4px',
                  borderRadius: '4px'
                }}
              >
                ×
              </button>
            </div>
            
            {/* Chat Content */}
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <ChatWindow />
            </div>
          </div>
        ) : (
          <button
            onClick={setChatVisible.setTrue}
            style={{
              width: '100%',
              height: '100%',
              border: 'none',
              borderRadius: '50%',
              backgroundColor: '#F87A53',
              color: 'white',
              fontSize: '28px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 16px rgba(255, 127, 0, 0.4)'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#c76344';
              // REMOVED: scale transform to eliminate zoom effect
              // e.currentTarget.style.transform = 'scale(1.1)';
              e.currentTarget.style.boxShadow = '0 6px 20px rgba(255, 127, 0, 0.6)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = '#F87A53';
              // REMOVED: scale transform to eliminate zoom effect
              // e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.boxShadow = '0 4px 16px rgba(255, 127, 0, 0.4)';
            }}
          >
            <img 
              src="/chatagent.png" 
              alt="AI Assistant" 
              style={{ 
                width: '40px', 
                height: '40px',
                filter: 'brightness(0) invert(1)'
              }} 
            />
          </button>
        )}
      </div>
    </>
  );
};

export default ModernSidebar;
