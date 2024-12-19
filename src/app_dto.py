from pydantic import BaseModel


class SMSRequest(BaseModel):
    sms: str


class URLRequest(BaseModel):
    url: str


class Response(BaseModel):
    type: str
    spam_percent: float

    def to_dict(self):
        return {
            "type": self.type,
            "spam_percent": self.spam_percent,
        }
