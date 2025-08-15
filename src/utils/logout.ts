import { API } from "./http";

export const LogOut = () => {
  return API.get("auth/logout");
};
