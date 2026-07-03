class RBIHIntegration:
    """Future MuleHunter/RBIH escalation adapter placeholder."""

    def submit_alerts(self, alerts):
        raise NotImplementedError("RBIH/MuleHunter submission is a future integration placeholder.")


class I4CIntegration:
    """Future I4C/NCRP reporting adapter placeholder."""

    def submit_suspicious_accounts(self, alerts):
        raise NotImplementedError("I4C/NCRP suspicious-account reporting is a future integration placeholder.")


class FederatedLearningIntegration:
    """Future cross-bank federated learning adapter placeholder."""

    def train_round(self):
        raise NotImplementedError("Federated learning is reserved for a future version.")
