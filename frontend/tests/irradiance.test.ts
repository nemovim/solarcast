import test from 'node:test';
import assert from 'node:assert/strict';
import { GET } from '../app/api/irradiance/route';

test('NASA climatology converts global and diffuse daily means to monthly energy', async () => {
  const original=globalThis.fetch;
  const values=Object.fromEntries(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'].map((month)=>[month,4]));
  const diffuse=Object.fromEntries(Object.keys(values).map((month)=>[month,2]));
  globalThis.fetch=async (url)=> {
    assert.match(String(url),/ALLSKY_SFC_SW_DWN%2CALLSKY_SFC_SW_DIFF/);
    return Response.json({parameters:{ALLSKY_SFC_SW_DWN:{units:'kW-hr/m^2/day'},ALLSKY_SFC_SW_DIFF:{units:'kW-hr/m^2/day'}},properties:{parameter:{ALLSKY_SFC_SW_DWN:values,ALLSKY_SFC_SW_DIFF:diffuse}}});
  };
  try {
    const response=await GET(new Request('http://localhost/api/irradiance?lat=35&lon=127'));
    assert.equal(response.status,200);
    const data=await response.json();
    assert.equal(data.monthly[0],124);
    assert.equal(data.diffuse[1],56);
  } finally { globalThis.fetch=original; }
});
