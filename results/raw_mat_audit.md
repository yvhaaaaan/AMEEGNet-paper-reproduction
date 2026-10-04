# Independent BCI IV 2a raw-data audit

The audit parsed the official MAT files directly with scipy.io and did not import the AMEEGNet data loader.

Window: [1.5, 6.0) s at 250 Hz; 22 EEG channels; 1125 samples per trial.

| Subject | T raw/cache | E raw/cache | T max abs diff | E max abs diff |
| --- | --- | --- | ---: | ---: |
| A01 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A02 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A03 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A04 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A05 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A06 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A07 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A08 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |
| A09 | [288, 22, 1125] / True | [288, 22, 1125] / True | 0 | 0 |

All nine subjects have 288 labelled trials per session and 72 trials per class per session.
The cached NPZ arrays and labels are compared in the original trial order; `arrays_bitwise_equal` is required to be true for both sessions of every subject.

## Source SHA-256

| File | SHA-256 |
| --- | --- |
| T (A01) | 054f02e70cf9c4ada1517e9b9864f45407939c1062c6793516585c6f511d0325 |
| E (A01) | 53d415f39c3d7b0c88b894d7b08d99bcdfe855ede63831d3691af1a45607fb62 |
| T (A02) | 5ddd5cb520b1692c3ba1363f48d98f58f0e46f3699ee50d749947950fc39db27 |
| E (A02) | d63c454005d3a9b41d8440629482e855afc823339bdd0b5721842a7ee9cc7b12 |
| T (A03) | 7e731ee8b681d5da6ecb11ae1d4e64b1653c7f15aad5d6b7620b25ce53141e80 |
| E (A03) | d4229267ec7624fa8bd3af5cbebac17f415f7c722de6cb676748f8cb3b717d97 |
| T (A04) | 15850d81b95fc88cc8b9589eb9b713d49fa071e28adaf32d675b3eaa30591d6e |
| E (A04) | 81916dff2c12997974ba50ffc311da006ea66e525010d010765f0047e771c86a |
| T (A05) | 77387d3b669f4ed9a7c1dac4dcba4c2c40c8910bae20fb961bb7cf5a94912950 |
| E (A05) | 8b357470865610c28b2f1d351beac247a56a856f02b2859d650736eb2ef77808 |
| T (A06) | 4dc3be1b0d60279134d1220323c73c68cf73799339a7fb224087a3c560a9a7e2 |
| E (A06) | bf67a40621b74b6af7a986c2f6edfff7fc2bbbca237aadd07b575893032998d1 |
| T (A07) | 43b6bbef0be78f0ac2b66cb2d9679091f1f5b7f0a5d4ebef73d2c7cc8e11aa96 |
| E (A07) | b9aaec73dcee002fab84ee98e938039a67bf6a3cbf4fc86d5d8df198cfe4c323 |
| T (A08) | 7a4b3bd602d5bc307d3f4527fca2cf076659e94aca584dd64f6286fd413a82f2 |
| E (A08) | 0eedbd89790c7d621c8eef68065ddecf80d437bbbcf60321d9253e2305f294f7 |
| T (A09) | b28d8a262c779c8cad9cc80ee6aa9c5691cfa6617c03befe490a090347ebd15c |
| E (A09) | 5d79649a42df9d51215def8ffbdaf1c3f76c54b88b9bbaae721e8c6fd972cc36 |

## Run structure

### A01
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A02
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A03
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A04
- T: run1=48 trials, run2=48 trials, run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A05
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A06
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A07
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A08
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
### A09
- T: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
- E: run3=48 trials, run4=48 trials, run5=48 trials, run6=48 trials, run7=48 trials, run8=48 trials
