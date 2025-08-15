import { useBoolean, useRequest } from "ahooks";
import { Button, Form, Input, Modal } from "antd";
import { FC, PropsWithChildren } from "react";

import { API } from "@/src/utils/http";

interface IProps {}

const ReportWildfre: FC<PropsWithChildren<IProps>> = () => {
  const [visible, setVisible] = useBoolean();
  const [form] = Form.useForm();

  const { runAsync, loading } = useRequest(
    (body: { latitude: number; longitude: number }) => {
      return API.post("/report", {
        data: body,
      });
    },
    {
      manual: true,
    },
  );
  const handlSubmit = async () => {
    const values = await form.validateFields();

    const res = await runAsync(values);

    setVisible.setFalse();
  };

  return (
    <>
      <Button onClick={setVisible.setTrue}>Report Wildfire</Button>
      <Modal loading={loading} open={visible} onCancel={setVisible.setFalse} onOk={handlSubmit}>
        <Form form={form} layout="vertical">
          <Form.Item required label="Longitude" name="longitude" rules={[{ required: true }]}>
            <Input required placeholder="Longitude" />
          </Form.Item>
          <Form.Item required label="Latitude" name="latitude" rules={[{ required: true }]}>
            <Input required placeholder="Latitude" />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

export default ReportWildfre;
