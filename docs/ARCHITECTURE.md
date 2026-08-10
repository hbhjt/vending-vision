# Vending Vision architecture

The runtime exposes one strict `vem.vision.v2` WebSocket protocol. Top-camera
presence, front-camera profile sampling, ambient light, departure, camera-role
maintenance, and health remain independent core capabilities.

Generated Fast and AI try-on work is represented by one attempt lifecycle and
uses attempt-scoped result references. The Fast attempt owns a unique
`try_on_attempt` front-camera acquisition lease only while acquiring its source
frame; profile sampling uses the `vision` lease. A result never travels through
the platform, MQTT, or managed-media cache.

The V2 bundle is authored in VEM Shared Contracts and copied into this
repository. The build checks JSON Schema, fixtures, generated Python models,
and manifest digest parity before packaging.
