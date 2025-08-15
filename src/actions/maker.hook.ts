import { useRequest } from "ahooks";

import { ReportWildfire, RoadClosure, User } from "@prisma/client";

import { API } from "../utils/http";

export type FirePoint = {
  lat: number;
  lng: number;
  raw: IRow;
};

export type IRoadClosure = RoadClosure & { User: User };
interface IRow {
  name: string;
  date: string;
  longitude: string;
  latitude: string;
  prettyname: string;
  acres: string;
  containper: string;
  xlo: string;
  xhi: string;
  ylo: string;
  yhi: string;
  irwinid: string;
  htb: string;
  modis: string;
  viirs: string;
  wfigs: string;
  firis: string;
  "fireguard ": string;
}

export const useMapMarkers = () => {
  const { data } = useRequest(() => {
    return API.get<FirePoint[]>("/fires");
  });

  return { markers: data || [] };
};

export const useReportMarkers = () => {
  const { data, run: refresh } = useRequest(() => {
    return API.get<ReportWildfire[]>("/report");
  });

  const { runAsync: delReport } = useRequest(
    (id: number) => {
      return API.delete<ReportWildfire>("/report", { data: { id } });
    },
    { manual: true, onSuccess: refresh },
  );

  return { reportMarkers: data || [], delReport };
};

export const useClosureMarkers = () => {
  const {
    data,
    run: getRoadClosure,
    loading: getLoading,
  } = useRequest(() => {
    return API.get<IRoadClosure[]>("/closure");
  });

  const { run: createClosure, loading: createLoading } = useRequest(
    (body: any) => {
      return API.post<RoadClosure>("/closure", { data: body });
    },
    {
      manual: true,
      onSuccess: getRoadClosure,
    },
  );

  const { runAsync: delClosure } = useRequest(
    (id: number) => {
      return API.delete<RoadClosure>("/closure", { data: { id } });
    },
    {
      manual: true,
      onSuccess: getRoadClosure,
    },
  );

  return { closureData: data || [], createClosure, getLoading, createLoading, delClosure };
};
