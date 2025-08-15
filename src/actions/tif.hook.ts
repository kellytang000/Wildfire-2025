import { useRequest } from "ahooks";
import { API } from "../utils/http";

export interface Option {
  value?: string | number | null;
  label: React.ReactNode;
  children?: Option[];
  isLeaf?: boolean;
}

export const useOptions = () => {
  const { data, run, loading, mutate, runAsync } = useRequest(
    (url = "", isFirstLevel = false) => {
      return API.get<Option[]>("/crawl/option", {
        data: { target: url, isFirstLevel },
      });
    },
    {
      manual: true,
    },
  );

  return { options: data || [], loading, getOptions: run, mutate, runAsync };
};
