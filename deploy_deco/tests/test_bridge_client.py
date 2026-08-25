from deploy_deco.bridge_client import build_websocket_uri


def test_local_address_gets_explicit_port():
    assert build_websocket_uri("127.0.0.1", 26421, None) == "ws://127.0.0.1:26421"


def test_https_tunnel_keeps_public_endpoint():
    assert (
        build_websocket_uri("https://robot.example.com/path", 26421, None)
        == "wss://robot.example.com/path"
    )
