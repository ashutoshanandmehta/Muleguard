from .models import EventData, NodeFeatures, EdgeFeatures

class FeatureService:
    def compute_delta(self, event: EventData, node_feats: NodeFeatures) -> EdgeFeatures:
        return EdgeFeatures(
            txn_freq=node_feats.account.values.get("txn_velocity_24h", 0.0),
            recency_decay=node_feats.account.values.get("recency_decay", 1.0),
            device_entropy=(node_feats.device.values.get("device_entropy", 0.0)
                            if node_feats.device else 0.0),
            session_risk=node_feats.account.values.get("ts_anomaly", 0.0)
        )
