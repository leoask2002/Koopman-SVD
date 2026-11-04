import numpy as np

from experiments.pendulum_baselines import fit_baselines


def toy():
    # An exact affine closure on the first sensor, with schedule-dependent sign.
    # Other sensors are not required for the target.
    rng = np.random.default_rng(27)
    damping = np.repeat([[.1], [2.8]], 3, axis=1)
    train = np.empty((2, 192, 4, 4))
    for i in range(2):
        c = rng.uniform(-.8,.8,size=192)
        for t in range(4):
            train[i,:,t] = np.column_stack((c,np.zeros(len(c)),rng.uniform(-.4,.4,len(c)),
                                             rng.normal(size=len(c))))
            c = (.75 if i == 0 else -.75)*c
    return train, damping


def test_conditioned_edmd_composes_unseen_sequence_without_target_fitting():
    tr,d = toy()
    models=fit_baselines(tr,d,view="partial")
    x=tr[0,:12,0]; target=[.1,2.8,.1]
    known=x[:,0,None]*np.cumprod([.75,-.75,.75])
    for name in ["edmd_damping","edmd_smoothed","edmd_rank4","edmd_rank8"]:
        np.testing.assert_allclose(models[name].predict(x,target),known,atol=.004)
    # Pooled model cannot distinguish the two candidate transitions.
    np.testing.assert_array_equal(models["edmd_pooled"].predict(x,target),
                                  models["edmd_pooled"].predict(x,[2.8]*3))
    assert models["edmd_rank4"].state_size <= 4
    assert models["edmd_rank8"].state_size <= 8


def test_direct_horizon_and_persistence_produce_bounded_scalar_output():
    tr,d=toy()
    models=fit_baselines(tr,d,view="partial")
    x=tr[0,:13,0]
    known=x[:,0,None]*.75**np.arange(1,4)
    np.testing.assert_allclose(models["direct_horizon"].predict(x,d[0]),known,atol=.003)
    p = models["persistence"].predict(x,d[0])
    np.testing.assert_array_equal(p,np.broadcast_to(x[:,0,None],p.shape))
    for model in models.values():
        pred = model.predict(x,d[0][:2])
        assert pred.shape == (13,2) and (abs(pred)<=1).all()


def test_harmonic_dictionary_does_not_read_velocity_in_partial_view():
    tr,d=toy()
    x=tr[0,:15,0]
    other=x.copy(); other[:,2:] += 2.5
    model=fit_baselines(tr,d,view="partial")["edmd_harmonic"]
    np.testing.assert_array_equal(model.predict(x,d[0]),model.predict(other,d[0]))
