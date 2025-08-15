'use client';

import {FC, PropsWithChildren} from 'react';

import {MapProvider} from '@/src/context/map.ctx';
interface IProps {}

const Layout: FC<PropsWithChildren<IProps>> = ({children}) => {
  return <MapProvider>{children}</MapProvider>;
};

export default Layout;
