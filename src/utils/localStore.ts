"use client";

const { parse, stringify } = JSON;

class LocalStore {
  private store!: Storage;
  type: "session" | "local";
  constructor(type: "session" | "local") {
    this.type = type;
  }
  init() {
    if (this.store) return;

    if (this.type == "local") {
      this.store = window.localStorage;
    } else {
      this.store = window.sessionStorage;
    }
  }

  set(key: string, value: any) {
    this.init();
    try {
      this.store.setItem(key, stringify(value));
    } catch (error) {
      console.warn(`store set出错:key:${key} value:${value}`);
    }
  }
  get(key: string) {
    this.init();
    const data = this.store.getItem(key);

    return data ? parse(data) : data;
  }
  remove(key: string): void {
    this.init();
    this.store.removeItem(key);
  }
  removeAll(): void {
    this.init();
    this.store.clear();
  }
}

export const localStore = new LocalStore("local");
export const sessionStore = new LocalStore("session");
