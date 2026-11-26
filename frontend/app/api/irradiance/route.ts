// NASA POWER monthly climatology, proxied so the browser can use a stable same-origin API.
export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const lon = Number(params.get('lon'));
  const lat = Number(params.get('lat'));
  if (!Number.isFinite(lon) || !Number.isFinite(lat) || lon < 123 || lon > 133 || lat < 32 || lat > 40) {
    return Response.json({ error: 'invalid_location' }, { status: 400 });
  }
  try {
    const url = new URL('https://power.larc.nasa.gov/api/temporal/climatology/point');
    url.search = new URLSearchParams({ parameters: 'ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF', community: 'RE', longitude: String(lon), latitude: String(lat), format: 'JSON' }).toString();
    const response = await fetch(url, { signal: AbortSignal.timeout(20000), next: { revalidate: 86400 } });
    if (!response.ok) throw new Error('POWER request failed');
    const body = await response.json();
    const units = body?.parameters?.ALLSKY_SFC_SW_DWN?.units;
    if (!['kW-hr/m^2/day', 'kWh/m^2/day'].includes(units)) throw new Error('unexpected units');
    if (!['kW-hr/m^2/day', 'kWh/m^2/day'].includes(body?.parameters?.ALLSKY_SFC_SW_DIFF?.units)) throw new Error('unexpected diffuse units');
    const source = body?.properties?.parameter?.ALLSKY_SFC_SW_DWN;
    const names = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
    const days = [31,28,31,30,31,30,31,31,30,31,30,31];
    const monthly = names.map((name, i) => Number(source?.[name]) * days[i]);
    const diffuseSource = body?.properties?.parameter?.ALLSKY_SFC_SW_DIFF;
    const diffuse = names.map((name, i) => Number(diffuseSource?.[name]) * days[i]);
    if (monthly.some((v) => !Number.isFinite(v) || v <= 0)) throw new Error('missing irradiance');
    if (diffuse.some((v,i) => !Number.isFinite(v) || v < 0 || v > monthly[i])) throw new Error('invalid diffuse irradiance');
    return Response.json({ monthly, diffuse });
  } catch {
    return Response.json({ error: 'irradiance_unavailable' }, { status: 502 });
  }
}
