import { useBoolean } from "ahooks";

export const useModal = () => {
  const [visible, setVisible] = useBoolean();

  return { visible, setVisible };
};
