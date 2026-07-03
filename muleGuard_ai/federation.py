class FederatedClient:
    """Local training client placeholder; real FL handled outside scoring path."""
    def send_update(self, model_update: dict) -> None:
        pass

    def receive_global(self) -> dict:
        return {}