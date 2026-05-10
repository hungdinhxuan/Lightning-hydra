M260429 = {
        'version': 'MDT_260226_lora_260429',
        'model_path':  '../pretrained/06feb26_xlsr_conformertcm_mdt_vad.pt',
        'adapter_path': '/NAS1_pretrained_lab/lora/29April26_xlsr_conformertcm_mdt_lora_replay_from_06feb26_xlsr_conformertcm_mdt_vad_bf16-mixed',
        'config_path': 'vocosig_model/configs/S_241214_tcm_conf-1_lora.yaml',
        'min_score': -5.8,
        'max_score': 5.26,
        'threshold': -0.62,
        'enable': True,
        'wrapper': 'VocoSigWrapper',
        'use_softmax': True,
        # config for the adversarial confidence
        'confidence_n_adv': 10,
        'confidence_max_amplitude': 14.0,
        'confidence_min_amplitude': 0.01,
        'confidence_is_feat_level': False,
        'confidence_config': {
                'eps': 1.0,
                'alpha': 0.01,
                'steps': 200,
                'random_start': True,
                'early_stop': True,
        },
        # config for XAI
        'xai_enable': True,
        'xai_num_samples': 30,
        'xai_asr_model': '../pretrained/whisper-base',
}