"""Small language model + regression head: text description of a run -> log10 final val loss."""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class SLMRegressor(nn.Module):
    def __init__(self, name="HuggingFaceTB/SmolLM2-135M"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, texts, device):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        h = self.lm(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)   # mean-pool over real tokens


def fit_predict(texts_tr, y_tr, texts_te, epochs=5, bs=16, lr=5e-5, name="HuggingFaceTB/SmolLM2-135M"):
    """Fine-tune on the train texts and return predictions for the test texts."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SLMRegressor(name).to(dev)
    mu, sd = y_tr.mean(), y_tr.std()                       # standardise the target
    y = torch.tensor((y_tr - mu) / sd, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(texts_tr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.mse_loss(model([texts_tr[j] for j in idx], dev), y[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep + 1}: last batch loss {loss.item():.3f}")
    model.eval()
    with torch.no_grad():
        preds = [model(texts_te[i:i + bs], dev).cpu() for i in range(0, len(texts_te), bs)]
    return torch.cat(preds).numpy() * sd + mu
