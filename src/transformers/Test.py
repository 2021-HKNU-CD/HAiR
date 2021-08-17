import cv2
from src.transformers.Transformer import Transformer, getTransformer
from src.transformers.ComponentFactory import *

alignerFactories = {
    'wing_based': AlignerWingFactory, # default
    # 'cv_based': AlignerFactory,
}

boundingBoxFactories = {
    'eye_distance_based': BoundingBoxFactory # default
}

balderFactories = {
    'model_G_5_170': BalderFactory, # default
    'model_G_5_170_retrained': Balder_5_170_Factory,
    'model_G_10_340': Balder_10_170_Factory,
}

def test(src_files=[], appearance_ref_files=[], ss_ref_files=[], tVariant=True, rVariant=True):
    # tVariant : True시 가능한 모든 Transformer 조합을 대상으로 테스트를 수행합니다.
    # rVariant : True시 가능한 모든 레퍼런스 조합을 대상으로 테스트를 수행 합니다.

    from itertools import product
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def getImage(filename):
        return cv2.imread('{}/../../data/{}.jpg'.format(BASE_DIR, filename))

    # 조합별 Transformer
    transformers = {'wing_based_eye_distance_based_model_G_5_170': getTransformer()}
    transformers['wing_based_eye_distance_based_model_G_5_170'].pass_through = False

    if tVariant:
        for a_key, bb_key, bd_key in product(alignerFactories.keys(),
                                             boundingBoxFactories.keys(),
                                             balderFactories.keys()):
            transformers['{}--{}--{}'.format(a_key, bb_key, bd_key)] = Transformer(boundingBoxFactories[bb_key],
                                                                                 alignerFactories[a_key],
                                                                                 balderFactories[bd_key])

    # 조합별 Reference
    appearance_refs = {file : getImage(file) for file in appearance_ref_files}
    appearance_refs['none'] = None

    shape_and_structure_refs = {file : getImage(file) for file in ss_ref_files}
    shape_and_structure_refs['none'] = None

    refs = {'none_{}_{}'.format(s_key, s_key) : [None, getImage(s_key)] for s_key in list(shape_and_structure_refs.keys())[:1]}

    if rVariant:
        for a_key, s_key, in product(appearance_refs.keys(), shape_and_structure_refs.keys()):
            refs['{}_{}_{}'.format(a_key, s_key, s_key)] = [appearance_refs[a_key],
                                                            shape_and_structure_refs[s_key]]

    if not os.path.isdir('results'):
        os.mkdir('results')

    import time
    start = time.time()
    for src_file in src_files:
        src = getImage(src_file)

        for r_key in refs.keys():
            appearance_ref, shape_and_structure_ref = refs[r_key]

            for t_key, transformer in transformers.items():
                result_dir = 'results/{}'.format(t_key)
                file = '{}/{}_{}.jpg'.format(result_dir, src_file, r_key)
                print(file, 'started')

                transformer.set_appearance_ref(appearance_ref)
                transformer.set_shape_ref(shape_and_structure_ref)
                transformer.set_structure_ref(shape_and_structure_ref)

                try:
                    result = transformer.transform(src)

                    if not os.path.isdir(result_dir):
                        os.mkdir(result_dir)
                    cv2.imwrite(file, result)
                except Exception as e:
                    print(e)

    print('total time :', time.time() - start)

test(src_files=['1'],
     appearance_ref_files=['lee'],
     ss_ref_files=['kim'],
     tVariant=False,
     rVariant=False)