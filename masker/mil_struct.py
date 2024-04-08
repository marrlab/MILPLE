_base_ = ['patients.py']


attributor = dict(
    type='VisionAttributor',

    layer='0.7.2.conv2',
    use_softmax=True,
   
    classifier=dict(source='mil', type='resnet34',pretrained='../models/model.pt'),
    
    feat_iba=dict(
        type='VisionFeatureIBA',
        input_or_output="output",
        active_neurons_threshold=0.01,
        initial_alpha=5.0),
    input_iba=dict(type='VisionInputIBA', initial_alpha=5.0, sigma=1.0),
    gan=dict(
        type='VisionWGAN',
        generator=dict(type='VisionGenerator'),
        discriminator=dict(type='VisionDiscriminator')))

estimation_cfg = dict(
    n_samples=15,
    verbose=False,
)

attribution_cfg = dict(
    feat_iba=dict(batch_size=10, beta=20, log_every_steps=-1),
    gan=dict(
        dataset_size=200,
        sub_dataset_size=20,
        lr=0.00005,
        batch_size=32,
        weight_clip=0.01,
        epochs=20,
        critic_iter=5,
        verbose=False),
    input_iba=dict(
        beta=20.0, opt_steps=60, lr=1.0, batch_size=10, log_every_steps=-1),
    feat_mask=dict(upscale=True, show=False),
    input_mask=dict(show=False))
