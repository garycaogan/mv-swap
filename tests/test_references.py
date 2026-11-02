"""Reference edits: fake APIMart, exact inputs, bounded redraw and paid task resume."""
import io
import json
import runpy
import sys
import tempfile
import unittest
import urllib.error
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

from mvswap import ai, config, finish, media, pipeline, references


class ReferenceEdits(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(mock.patch.object(config, 'apimart_key', return_value='fake-key'))
        self.stack.enter_context(mock.patch.object(config, 'vision_key', return_value='fake-key'))
        self.stack.enter_context(mock.patch.object(ai, 'upload', side_effect=AssertionError('upload forbidden')))
        self.stack.enter_context(mock.patch.object(ai.time, 'sleep'))
        self.pic = self.root / 'primary.png'
        self.first = self.root / 'first.jpg'
        for p in (self.pic, self.first): Image.new('RGB', (64, 96), 'blue').save(p)
        self.character = {'id':'P1', 'name':'千咲', 'picture':str(self.pic), 'desc':'black hair, red ribbons, navy shirt, black skirt',
                          'src_desc':'the woman', 'references':[{'picture':str(self.pic)}]}
        self.geometry = {'framing':'medium','head_top_y':.19}
        self.job = pipeline.Job(self.root / 'job')
        self.shot = {'id':'s001','kind':'swap','people':['P1'],'framing':'medium','geometry':{'P1':self.geometry},
                     'f0':24,'frames':72,'gen_frames':90}
        self.job.state = {'args':{'source':'source.mp4','cast':['woman=千咲'],'start':0,'end':4,'ref_mode':'neutral'},
                          'prepared':{'mv':'source.mp4','frames':96,'width':1080,'height':1920},
                          'cast':[self.character], 'shots':[self.shot]}

    def good(self):
        return {k:True for k in ('same_identity','hair_kept','ornaments_kept','eyes_kept','outfit_kept','framing_matches',
                                'headroom_matches','anatomy_valid','front_facing','arms_at_sides',
                                'pose_matches_first','camera_matches_first','background_matches_first')}

    def fake_image(self, out, prompt, refs, **kwargs):
        Image.new('RGB',(64,96),'red').save(out)
        kwargs['state'].update(task='fake-image-task', cny=.32, status='done')
        kwargs['checkpoint']()
        return .32

    def test_neutral_uses_only_primary_never_source_photo_and_preserves_identity_description(self):
        with mock.patch.object(ai,'image',side_effect=self.fake_image) as edit, \
                mock.patch.object(ai,'ask_json',return_value=self.good()), mock.patch.object(media,'keyframes') as extract:
            pipeline.stage_references(self.job)
        self.assertEqual(edit.call_args.args[2],[self.pic])
        self.assertIn('BOTH ARMS',edit.call_args.args[1])
        self.assertIn('19%',edit.call_args.args[1])
        self.assertNotIn('black skirt',edit.call_args.args[1])
        extract.assert_not_called()
        self.assertIn('red ribbons',self.shot['references']['P1']['desc'])

    def test_keyframe_inputs_are_exact_first_frame_then_primary(self):
        self.job.state['args']['ref_mode']='keyframe'
        with mock.patch.object(ai,'image',side_effect=self.fake_image) as edit, mock.patch.object(ai,'ask_json',return_value=self.good()), \
                mock.patch.object(media,'keyframes',return_value=[self.first]) as extract:
            pipeline.stage_references(self.job)
        self.assertEqual(extract.call_args.args[1],[24])
        self.assertEqual(edit.call_args.args[2],[self.first,self.pic])
        self.assertIn('POSE from image 1, IDENTITY from image 2',edit.call_args.args[1])
        self.assertIn('replace ONLY the woman; every other person stays unchanged',edit.call_args.args[1])

    def test_bad_neutral_pose_redraws_once_and_keeps_both_costs(self):
        bad = {**self.good(),'arms_at_sides':False,'issue':'hand touches hair'}
        with mock.patch.object(ai,'image',side_effect=self.fake_image) as edit, mock.patch.object(ai,'ask_json',side_effect=[bad,self.good()]):
            pipeline.stage_references(self.job)
        self.assertEqual(edit.call_count,2)
        self.assertIn('hand touches hair',edit.call_args.args[1])
        self.assertEqual(self.shot['references']['P1']['edit_attempt'],2)
        self.assertAlmostEqual(sum(a['cny'] for a in self.shot['reference_attempts']['P1']),.64)
        report=finish.report(self.job.state,self.job.dir).read_text()
        self.assertIn('参考图编辑 ¥0.64',report)

    def test_two_bad_edits_stop_and_resume_does_not_buy_a_third(self):
        bad={**self.good(),'anatomy_valid':False}
        with mock.patch.object(ai,'image',side_effect=self.fake_image) as edit, mock.patch.object(ai,'ask_json',return_value=bad):
            with self.assertRaisesRegex(RuntimeError,'两次'): pipeline.stage_references(self.job)
            with self.assertRaisesRegex(RuntimeError,'两次'): pipeline.stage_references(pipeline.Job(self.job.dir))
        self.assertEqual(edit.call_count,2)

    def test_manual_rejection_of_ai_passed_image_allows_only_one_redraw(self):
        with mock.patch.object(ai,'image',side_effect=self.fake_image),mock.patch.object(ai,'ask_json',return_value=self.good()):
            pipeline.stage_references(self.job)
            self.shot['reference_attempts']['P1'][0]['manual_pass']=False
            self.shot.pop('references')
            pipeline.stage_references(self.job)
        self.assertEqual(self.shot['references']['P1']['edit_attempt'],2)

    def test_successful_reference_is_reused_without_paid_calls(self):
        with mock.patch.object(ai,'image',side_effect=self.fake_image),mock.patch.object(ai,'ask_json',return_value=self.good()):
            pipeline.stage_references(self.job)
        with mock.patch.object(ai,'image',side_effect=AssertionError('paid again')),mock.patch.object(ai,'ask_json',side_effect=AssertionError('vision again')):
            pipeline.stage_references(pipeline.Job(self.job.dir))

    def test_refs_only_stops_before_any_video_or_upscale(self):
        self.job.save()
        with ExitStack() as stack:
            for name in ('stage_prepare','stage_shots','stage_cast','stage_label','stage_backdrops'):
                stack.enter_context(mock.patch.object(pipeline,name))
            stack.enter_context(mock.patch.object(ai,'image',side_effect=self.fake_image))
            stack.enter_context(mock.patch.object(ai,'ask_json',return_value=self.good()))
            generate=stack.enter_context(mock.patch.object(pipeline,'stage_generate'))
            upscale=stack.enter_context(mock.patch.object(pipeline,'stage_upscale'))
            result=pipeline.run(self.job.dir,None,[],refs_only=True,ref_mode='keyframe')
        self.assertEqual(result,{'references_ready':True})
        generate.assert_not_called();upscale.assert_not_called()
        self.assertEqual(pipeline.Job(self.job.dir).state['args']['ref_mode'],'neutral')
        self.assertGreater(pipeline.Job(self.job.dir).state['t_reference_done'],0)

    def test_policy_refusal_is_preserved_and_never_retried(self):
        self.job.state['args']['ref_mode']='keyframe';self.job.save()
        refusal='Cannot edit this real person photograph: policy rejection.'
        def reject(*args,**kwargs):
            kwargs['state'].update(status='failed',error=refusal,cny=.05)
            kwargs['checkpoint']()
            raise RuntimeError(refusal)
        with ExitStack() as stack:
            for name in ('stage_prepare','stage_shots','stage_cast','stage_label','stage_backdrops'):
                stack.enter_context(mock.patch.object(pipeline,name))
            stack.enter_context(mock.patch.object(media,'keyframes',return_value=[self.first]))
            edit=stack.enter_context(mock.patch.object(ai,'image',side_effect=reject))
            generate=stack.enter_context(mock.patch.object(pipeline,'stage_generate'))
            for _ in range(2):
                with self.assertRaisesRegex(SystemExit,'policy rejection'):pipeline.run(self.job.dir,None,[])
        self.assertEqual(edit.call_count,1);generate.assert_not_called()
        saved=pipeline.Job(self.job.dir).state
        self.assertEqual(saved['reference_blocked'],refusal)
        self.assertIn(refusal,(self.job.dir/'report.md').read_text())

    def test_image_billing_and_task_survive_download_failure_without_resubmission(self):
        state={};out=self.root/'edited.png';buffer=io.BytesIO();Image.new('RGB',(8,8)).save(buffer,format='PNG')
        result={'data':{'status':'completed','cost':.05,'result':{'images':[{'url':['https://example.invalid/image.png']}]}}}
        saved=[]
        with mock.patch.object(config,'apimart_key',return_value='fake-key'),mock.patch.object(ai,'upload',return_value='https://example.invalid/ref'), \
                mock.patch.object(ai,'_req',side_effect=[{'data':[{'task_id':'edit-1'}]},result,result]) as request:
            with mock.patch('urllib.request.urlopen',side_effect=OSError('download failed')):
                with self.assertRaises(OSError):ai.image(out,'edit',[self.pic],state=state,checkpoint=lambda:saved.append(dict(state)))
            self.assertEqual(state['task'],'edit-1');self.assertAlmostEqual(state['cny'],.36)
            with mock.patch('urllib.request.urlopen',return_value=io.BytesIO(buffer.getvalue())):
                self.assertAlmostEqual(ai.image(out,'edit',[self.pic],state=state),.36)
        self.assertEqual(sum(call.args[2]=='POST' for call in request.call_args_list),1)
        self.assertFalse(request.call_args_list[0].kwargs['retry'])
        self.assertTrue(any(row.get('cny')==.36 for row in saved))

    def test_unknown_creation_result_blocks_repeat_post(self):
        state={}
        with mock.patch.object(config,'apimart_key',return_value='fake-key'),mock.patch.object(ai,'upload',return_value='ref'), \
                mock.patch.object(ai,'_req',side_effect=TimeoutError('timeout')) as request:
            for _ in range(2):
                with self.assertRaises((TimeoutError,RuntimeError)):ai.image(self.root/'out.png','edit',[self.pic],state=state)
        self.assertEqual(request.call_count,1)

    def test_http_refusal_keeps_original_body_and_does_not_retry(self):
        refusal='{"error":"real-person editing refused"}'
        error=urllib.error.HTTPError('https://example.invalid',400,'bad',{},io.BytesIO(refusal.encode()))
        with mock.patch('urllib.request.urlopen',side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError,'real-person editing refused'):
                ai._req('https://example.invalid','fake-key','POST','/images/generations',{},retry=False)
        request.assert_called_once()

    def test_cli_passes_reference_mode_and_refs_only(self):
        entry=Path(__file__).resolve().parents[1]/'mvswap.py'
        with mock.patch.object(sys,'argv',[str(entry),'run',str(self.job.dir),'--ref-mode','neutral','--refs-only']), \
                mock.patch.object(sys,'path',sys.path.copy()),mock.patch.object(sys,'stdout',io.StringIO()),mock.patch.object(sys,'stderr',io.StringIO()), \
                mock.patch.object(pipeline,'run') as run:
            runpy.run_path(str(entry),run_name='__main__')
        self.assertEqual(run.call_args.kwargs['ref_mode'],'neutral');self.assertTrue(run.call_args.kwargs['refs_only'])

    def test_new_job_uses_configured_default_and_resume_keeps_it(self):
        path=self.root/'new-job'
        with mock.patch.object(config,'DEFAULT_REF_MODE','keyframe'),mock.patch.object(pipeline,'_run',return_value={}):
            pipeline.run(path,'source.mp4',['woman=primary.png'])
            pipeline.run(path,None,[],ref_mode='crop')
        self.assertEqual(pipeline.Job(path).state['args']['ref_mode'],'keyframe')
