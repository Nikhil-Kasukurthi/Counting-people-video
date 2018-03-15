 off
    proxy / localhost:7003 {
        websocket
    }
}
