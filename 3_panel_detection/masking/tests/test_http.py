import http.client
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
from http.server import ThreadingHTTPServer
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Store, mask_png
from server import make_handler
from predictions import Predictions


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        assignment = root/'assignments.json'
        assignment.write_text(json.dumps({'images':[{'id':'1_1','file':'1_1.jpeg','part':1,'width':8,'height':8}]}), 'utf-8')
        self.store = Store(assignment, root, root/'work')
        prediction_root=root/'predictions'
        (prediction_root/'masks').mkdir(parents=True)
        (prediction_root/'summary.csv').write_text('key,width,height,instances,mask_path\n1_1,8,8,0,masks/1_1.png\n',encoding='utf-8-sig')
        (prediction_root/'masks/1_1.png').write_bytes(mask_png(8,8,[]))
        handler = make_handler(self.store, Predictions(prediction_root))
        handler.log_message = lambda *args: None
        self.server = ThreadingHTTPServer(('127.0.0.1',0), handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, data=None, headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        try:
            conn.request(method,path,json.dumps(data) if data is not None else None,headers or {})
            response=conn.getresponse()
            return response.status,response.read(),dict(response.getheaders())
        finally:
            conn.close()

    def test_routes_save_reopen_conflict_and_export(self):
        status,body,headers=self.request('GET','/')
        self.assertEqual(status,200)
        self.assertIn(b'id="canvas"',body)
        self.assertIn('Content-Security-Policy',headers)
        record=self.store.blank('1_1')
        record.update(status='empty')
        self.assertEqual(self.request('POST','/api/records/1_1',record)[0],200)
        self.assertEqual(json.loads(self.request('GET','/api/records/1_1')[1])['revision'],1)
        self.assertEqual(self.request('POST','/api/records/1_1',record)[0],409)
        status,body,headers=self.request('GET','/api/export/1')
        self.assertEqual(status,200)
        self.assertTrue(body.startswith(b'PK'))

    def test_reject_external_origin_unknown_image_and_invalid_body(self):
        record=self.store.blank('1_1')
        self.assertEqual(self.request('POST','/api/records/1_1',record,{'Origin':'https://example.com'})[0],400)
        self.assertEqual(self.request('GET','/api/records/../../outside')[0],400)
        self.assertEqual(self.request('POST','/api/records/1_1',[])[0],400)
        self.assertFalse(self.store.workspace.exists())

    def test_prediction_overlay_is_read_only_and_zero_is_available(self):
        status,body,_=self.request('GET','/api/predictions/1_1')
        self.assertEqual(status,200)
        data=json.loads(body)
        self.assertTrue(data['available'])
        self.assertEqual(data['instances'],0)
        self.assertEqual(self.request('GET',data['url'])[1],mask_png(8,8,[]))
        self.assertEqual(self.store.read('1_1')['polygons'],[])
        self.assertFalse(self.store.workspace.exists())
        (Path(self.temp.name)/'predictions/masks/1_1.png').unlink()
        self.assertFalse(json.loads(self.request('GET','/api/predictions/1_1')[1])['available'])


if __name__ == '__main__':
    unittest.main()
