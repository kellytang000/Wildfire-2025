'use client'

import { localStore } from './localStore';

const KEY = 'authorization';

export function setToken(value: any) {
  localStore.set(KEY, value);
}

export function getToken() {
  return localStore.get(KEY);
}

export function removeToken() {
  localStore.remove(KEY);
}
