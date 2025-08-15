-- CreateTable
CREATE TABLE "User" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "username" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "role" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "RoadClosure" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "lat" REAL NOT NULL,
    "lng" REAL NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "userId" INTEGER NOT NULL,
    CONSTRAINT "RoadClosure_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Shelter" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "shelterId" TEXT NOT NULL,
    "lat" REAL NOT NULL,
    "lng" REAL NOT NULL,
    "capacity" INTEGER NOT NULL,
    "hexId" TEXT NOT NULL,
    "region" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "ReportWildfire" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "lat" REAL NOT NULL,
    "lng" REAL NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "userId" INTEGER NOT NULL,
    CONSTRAINT "ReportWildfire_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "CustomFire" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "objectId" BIGINT NOT NULL,
    "lat" REAL NOT NULL,
    "lng" REAL NOT NULL,
    "type" TEXT NOT NULL,
    "shapeArea" REAL NOT NULL,
    "shapeLength" REAL NOT NULL,
    "geometry" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateIndex
CREATE UNIQUE INDEX "User_username_key" ON "User"("username");

-- CreateIndex
CREATE UNIQUE INDEX "Shelter_shelterId_key" ON "Shelter"("shelterId");
