"""Check that sufficient-statistic transport equals direct weighted-path transport."""
import numpy as np
from experiments.pendulum_neural import PrefixSpectralForecast, fit_neural
from koopman_svd import TimeIndexedKoopmanSVD


def test_prefix_mixture_agrees_with_integer_weighted_paths():
    rng = np.random.default_rng(109)
    paths = rng.uniform(-1., 1., size=(2,20,4,3))
    damping = np.array([[.2]*3,[.8]*3])
    # At schedule 0.2, Gaussian prefix weights have exact ratio 2:1.
    bw = .6 / np.sqrt(2*np.log(2))
    model = PrefixSpectralForecast(paths,damping,rank=3,model=None,name="sensor_spectral",bandwidth=bw)
    repeated = np.concatenate((paths[0],paths[0],paths[1]))
    expected = TimeIndexedKoopmanSVD.fit(repeated,rank=3,rcond=1e-9)
    x = rng.uniform(-1.,1.,(10,3))
    y = np.column_stack([expected.predict(x,horizon=h,observable=lambda z:z[:,0])[:,0]
                         for h in (1,2,3)])
    np.testing.assert_allclose(model.predict(x,damping[0]),np.clip(y,-1,1),atol=8e-13)


def test_zero_updates_match_frozen_and_constant():
    rng = np.random.default_rng(190)
    obs = rng.uniform(-1.,1.,size=(2,30,4,3))
    obs[:,:,:,0] = .31
    d = np.array([[.2,.4,.7],[.7,.6,.4]])
    models = fit_neural(obs,d,obs[:,:10],d,seed=19,matched_rank=3,steps=0,width=4)
    for model in models.values():
        np.testing.assert_allclose(model.predict(obs[0,:8,0],d[1]),.31,atol=3e-12)
    assert np.array_equal(models["neural"].model.raw_functions(obs[1,:3,1]),
                          models["frozen"].model.raw_functions(obs[1,:3,1]))
