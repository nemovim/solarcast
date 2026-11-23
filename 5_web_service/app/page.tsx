import Workspace from './workspace';
import { connection } from 'next/server';

export default async function Home() {
  await connection();
  return <Workspace vworldEnabled={Boolean(process.env.VWORLD_KEY?.trim())} />;
}
