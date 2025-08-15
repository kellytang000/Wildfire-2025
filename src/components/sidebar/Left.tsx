"use client";

import { useBoolean, useRequest } from "ahooks";
import { Button, Card, Cascader, Form, Input, Modal, Spin } from "antd";
import { FC, PropsWithChildren, useEffect, useMemo, useState } from "react";

import ChatWindow from "../ai/ChatWindow";
import { UserLocationPanel } from "../UserLocationPanel";

import { Option, useOptions } from "@/src/actions/tif.hook";
import { useMapContext } from "@/src/context/map.ctx";
import { API } from "@/src/utils/http";
import { useUser } from "@/src/hooks/useUser";
import { useEscape } from "@/src/actions/closure.hook";
import ShelterManager from "@/src/components/ShelterManager";
import { useRouter } from "next/navigation";

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
const Left: FC<PropsWithChildren<IProps>> = () => {
  const router = useRouter();
  const { options, getOptions, loading: firstLoading } = useOptions();
  const { loading, runAsync: loadOption } = useOptions();
  const [values, setValues] = useState<(string | number | null)[]>([]);
  const [list, setList] = useState<Option[]>([]);
  const {
    currentMarker,
    setOverlayData,
    setClosureMode,
    closureMode,
    route,
    setRoute,
    showShelters,
    setShowShelters,
    setTifUrl,
    refreshShelters,
    setCenter,
  } = useMapContext();
  const { user, logout } = useUser();
  const [showShelterManager, setShowShelterManager] = useBoolean();

  const [isModalOpen, setIsModalOpen] = useBoolean();
  const { createEscape, loading: loadingEscape } = useEscape();
  const [chatVisible, setChatVisible] = useBoolean();

  const [form] = Form.useForm();

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
    const lastItem = selectedOptions[selectedOptions.length - 1];

    if (lastItem) {
      const target = selectedOptions.map((item) => item.value).join("");

      loadOption(`${currentMarker?.raw!.name}/${target}`).then((res) => {
        lastItem.children = res;
        setList([...list]);
      });
    }
  };

  const onChange = async (vals: (string | number | null)[]) => {
    setValues(vals);
    const tifUrl = `${currentMarker?.raw!.name}/${vals.join("")}`;

    setTifUrl(tifUrl);
    setOverlayData(null);
    //加载tif文件
    loadTif(tifUrl, currentMarker!.lat, currentMarker!.lng);
  };
  //
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

  useMemo(() => {
    if (currentMarker) {
      setValues([]);
      getOptions(`${currentMarker.raw.name}/`, true);
    }
  }, [currentMarker]);

  useEffect(() => {
    if (options.length) {
      setList(options);
    }
  }, [options]);

  return (
    <div className="absolute top-28 left-2">
      {showShelterManager ? (
        <div className="fixed inset-0 bg-white z-50 overflow-auto">
          <ShelterManager />
          <Button className="fixed top-4 right-4 z-60" onClick={() => setShowShelterManager.setFalse()}>
            Close
          </Button>
        </div>
      ) : (
        <Card
          extra={loadingTif && <Spin />}
          style={{ width: 300 }}
          title="Fire spread forecast"
          variant="borderless"
        >
          <div className="flex flex-col gap-y-5">
            <Cascader
              className="w-full!"
              loadData={loadData}
              loading={firstLoading || loading}
              options={list}
              placeholder="Please select"
              value={values}
              onChange={onChange}
            />
            {user && user.role === "ADMIN" && (
              <>
                <Button type={closureMode ? "primary" : "default"} onClick={() => setClosureMode.toggle()}>
                  {closureMode ? "Done" : "Set closure"}
                </Button>
                <Button
                  type={showShelterManager ? "primary" : "default"}
                  onClick={() => setShowShelterManager.toggle()}
                >
                  {showShelterManager ? "Close Shelter Manager" : "Manage Shelters"}
                </Button>
                <Button
                  type="default"
                  onClick={() => router.push('/admin/custom-fires')}
                >
                  Manage Custom Fires
                </Button>
              </>
            )}
            <Button onClick={setChatVisible.setTrue}>Ai Chat</Button>
            <div className="flex flex-col gap-2">
              <Button type={showShelters ? "primary" : "default"} onClick={() => setShowShelters(!showShelters)}>
                {showShelters ? "Hide shelters" : "Show shelters"}
              </Button>
              {showShelters && (
                <>
                  <Button type="default" onClick={() => setShowShelters(false)}>
                    Clear shelters
                  </Button>
                  <Button type="default" onClick={() => refreshShelters()}>
                    Refresh shelters
                  </Button>
                  <Button type="default" onClick={() => {
                    // Just show LA shelter info, don't change map center
                    console.log('LA shelter button clicked - display only');
                  }}>
                    Go to LA shelter
                  </Button>
                  <Button type="default" onClick={() => {
                    // Just show Squamish shelter info, don't change map center  
                    console.log('Squamish shelter button clicked - display only');
                  }}>
                    Go to Squamish shelter
                  </Button>
                </>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <Button type="primary" onClick={setIsModalOpen.setTrue}>
                Planning the path
              </Button>
              {route && (
                <>
                  <div className="text-sm text-gray-600">
                    Route: {route.n_points} points, {route.cost_s}s
                  </div>
                  <Button type="default" onClick={() => setRoute(null)}>
                    Clear route
                  </Button>
                </>
              )}
            </div>
            {user && (
              <>
                <Button type="primary" onClick={logout}>
                  Log out
                </Button>
              </>
            )}
          </div>
          {/* User Location Panel */}
          {user && (
            <div style={{ marginTop: 16 }}>
              <UserLocationPanel />
            </div>
          )}
          <Modal
            closable={true}
            okButtonProps={{
              loading: loadingEscape,
            }}
            open={isModalOpen}
            title="Please enter your latitude and longitude"
            onCancel={setIsModalOpen.setFalse}
            onOk={handleClosure}
          >
            <Form form={form} layout="vertical">
              <Form.Item required label="Latitude" name="latitude" rules={[{ required: true }]}>
                <Input required placeholder="Latitude" />
              </Form.Item>
              <Form.Item required label="Longitude" name="longitude" rules={[{ required: true }]}>
                <Input required placeholder="Longitude" />
              </Form.Item>
            </Form>
          </Modal>
        </Card>
      )}
      <Modal 
        footer={null} 
        open={chatVisible} 
        title="Ai Chat" 
        onCancel={setChatVisible.setFalse}
        transitionName=""
        maskTransitionName=""
      >
        <ChatWindow />
      </Modal>
    </div>
  );
};

export default Left;
