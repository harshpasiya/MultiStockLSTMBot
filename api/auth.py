"""
╔══════════════════════════════════════════════════════════════════════════╗
║         G.O.D.S E.Y.E — Authentication & Token Management               ║
║         Project : MultiStockLSTMBot                                      ║
║         File    : api/auth.py                                            ║
║         Phase   : 5 — Subscriber Infrastructure                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import uuid
import hashlib
import secrets
import string
import psycopg2
import psycopg2.extras

from datetime    import datetime, timedelta, timezone
from enum        import Enum
from dataclasses import dataclass, field
from typing      import Optional, Dict, List
from loguru      import logger
from dotenv      import load_dotenv

load_dotenv()

DB_URL         = os.getenv("TIMESCALE_URL",
                           "postgresql://godseye_user:godseye_pass@localhost:5433/godseye")
JWT_SECRET     = os.getenv("JWT_SECRET",     secrets.token_hex(32))
JWT_ALGORITHM  = "HS256"
JWT_EXPIRE_HRS = 24
AES_KEY        = os.getenv("AES_ENCRYPT_KEY", secrets.token_hex(16))[:32]

try:
    from jose import jwt, JWTError
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("python-jose not installed. JWT auth disabled. "
                   "Install: pip install python-jose[cryptography]")

try:
    from passlib.context import CryptContext
    pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("passlib not installed. Using SHA256 fallback.")

try:
    from cryptography.fernet import Fernet
    import base64
    _fernet_key = base64.urlsafe_b64encode(AES_KEY.encode().ljust(32)[:32])
    fernet      = Fernet(_fernet_key)
    AES_AVAILABLE = True
except ImportError:
    AES_AVAILABLE = False
    logger.warning("cryptography not installed. Broker tokens stored as plaintext.")


class UserRole(str, Enum):
    SUPER_ADMIN  = "super_admin"
    BROKER       = "broker"
    DISTRIBUTOR  = "distributor"
    SUBSCRIBER   = "subscriber"


class BrokerProvider(str, Enum):
    ZERODHA = "zerodha"
    UPSTOX  = "upstox"
    ANGEL   = "angel"


class SubscriptionPlan(str, Enum):
    FREE     = "free"
    BASIC    = "basic"
    PRO      = "pro"
    ELITE    = "elite"


class TokenType(str, Enum):
    ACCESS   = "access"
    REFRESH  = "refresh"
    API_KEY  = "api_key"


@dataclass
class User:
    user_id       : str
    email         : str
    role          : UserRole
    plan          : SubscriptionPlan  = SubscriptionPlan.FREE
    is_active     : bool              = True
    referral_code : str               = ""
    referred_by   : str               = ""
    distributor_id: str               = ""
    broker_id     : str               = ""
    created_at    : datetime          = field(default_factory=datetime.now)
    metadata      : Dict              = field(default_factory=dict)


@dataclass
class BrokerToken:
    token_id      : str
    user_id       : str
    provider      : BrokerProvider
    access_token  : str
    refresh_token : str               = ""
    expires_at    : Optional[datetime]= None
    account_id    : str               = ""
    is_active     : bool              = True


@dataclass
class APIKey:
    key_id        : str
    user_id       : str
    key_hash      : str
    name          : str               = ""
    scopes        : List[str]         = field(default_factory=list)
    is_active     : bool              = True
    last_used_at  : Optional[datetime]= None
    created_at    : datetime          = field(default_factory=datetime.now)


@dataclass
class ReferralCode:
    code          : str
    owner_id      : str
    owner_role    : UserRole
    uses          : int               = 0
    max_uses      : int               = 0
    commission_pct: float             = 0.0
    is_active     : bool              = True
    created_at    : datetime          = field(default_factory=datetime.now)
    expires_at    : Optional[datetime]= None


@dataclass
class AuthToken:
    access_token  : str
    token_type    : str      = "bearer"
    expires_in    : int      = JWT_EXPIRE_HRS * 3600
    user_id       : str      = ""
    role          : str      = ""


def hash_password(password: str) -> str:
    if BCRYPT_AVAILABLE:
        return pwd_context.hash(password)
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    if BCRYPT_AVAILABLE:
        try:
            return pwd_context.verify(plain, hashed)
        except Exception:
            pass
    return hashlib.sha256(plain.encode()).hexdigest() == hashed


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key(prefix: str = "ge") -> str:
    alphabet = string.ascii_letters + string.digits
    random   = "".join(secrets.choice(alphabet) for _ in range(32))
    return f"{prefix}_{random}"


def generate_referral_code(name: str = "", length: int = 8) -> str:
    if name:
        prefix = name.upper()[:5].replace(" ", "")
        suffix = "".join(secrets.choice(string.digits) for _ in range(4))
        return f"{prefix}{suffix}"
    chars  = string.ascii_uppercase + string.digits
    return "GE_" + "".join(secrets.choice(chars) for _ in range(length))


def create_jwt_token(
    user_id  : str,
    role     : UserRole,
    plan     : SubscriptionPlan = SubscriptionPlan.FREE,
    expire_hrs: int = JWT_EXPIRE_HRS,
) -> AuthToken:
    if not JWT_AVAILABLE:
        token = f"NONJWT_{user_id}_{secrets.token_hex(16)}"
        return AuthToken(access_token=token, user_id=user_id, role=role.value)

    # FIX: use timezone-aware UTC datetime — datetime.utcnow() is
    # deprecated and scheduled for removal in future Python versions.
    expire  = datetime.now(timezone.utc) + timedelta(hours=expire_hrs)
    payload = {
        "sub"  : user_id,
        "role" : role.value,
        "plan" : plan.value,
        "exp"  : expire,
        "iat"  : datetime.now(timezone.utc),
        "jti"  : str(uuid.uuid4()),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return AuthToken(
        access_token = token,
        expires_in   = expire_hrs * 3600,
        user_id      = user_id,
        role         = role.value,
    )


def verify_jwt_token(token: str) -> Optional[Dict]:
    if not JWT_AVAILABLE:
        if token.startswith("NONJWT_"):
            parts = token.split("_")
            return {"sub": parts[1], "role": "subscriber"} if len(parts) >= 2 else None
        return None

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except Exception as e:
        logger.debug(f"JWT verification failed: {e}")
        return None


def encrypt_token(raw_token: str) -> str:
    if AES_AVAILABLE:
        return fernet.encrypt(raw_token.encode()).decode()
    import base64
    return base64.b64encode(raw_token.encode()).decode()


def decrypt_token(encrypted_token: str) -> str:
    if AES_AVAILABLE:
        return fernet.decrypt(encrypted_token.encode()).decode()
    import base64
    return base64.b64decode(encrypted_token.encode()).decode()


class AuthManager:
    def __init__(self, conn=None):
        self._conn      = conn
        self._own_conn  = conn is None
        self._ensure_tables()

    def _get_conn(self):
        if self._conn and not self._conn.closed:
            return self._conn
        return psycopg2.connect(DB_URL)

    def _ensure_tables(self):
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id        VARCHAR(36)  PRIMARY KEY,
                        email          VARCHAR(200) UNIQUE NOT NULL,
                        password_hash  VARCHAR(200) NOT NULL,
                        role           VARCHAR(20)  NOT NULL DEFAULT 'subscriber',
                        plan           VARCHAR(20)  NOT NULL DEFAULT 'free',
                        is_active      BOOLEAN      NOT NULL DEFAULT TRUE,
                        referral_code  VARCHAR(20)  UNIQUE,
                        referred_by    VARCHAR(20),
                        distributor_id VARCHAR(36),
                        broker_id      VARCHAR(36),
                        metadata       JSONB        DEFAULT '{}',
                        created_at     TIMESTAMP    DEFAULT NOW(),
                        updated_at     TIMESTAMP    DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS broker_tokens (
                        token_id       VARCHAR(36)  PRIMARY KEY,
                        user_id        VARCHAR(36)  NOT NULL REFERENCES users(user_id),
                        provider       VARCHAR(20)  NOT NULL,
                        access_token   TEXT         NOT NULL,
                        refresh_token  TEXT,
                        expires_at     TIMESTAMP,
                        account_id     VARCHAR(50),
                        is_active      BOOLEAN      DEFAULT TRUE,
                        created_at     TIMESTAMP    DEFAULT NOW(),
                        updated_at     TIMESTAMP    DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id         VARCHAR(36)  PRIMARY KEY,
                        user_id        VARCHAR(36)  NOT NULL REFERENCES users(user_id),
                        key_hash       VARCHAR(200) NOT NULL,
                        name           VARCHAR(100),
                        scopes         TEXT[]       DEFAULT '{}',
                        is_active      BOOLEAN      DEFAULT TRUE,
                        last_used_at   TIMESTAMP,
                        created_at     TIMESTAMP    DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS referral_codes (
                        code           VARCHAR(20)  PRIMARY KEY,
                        owner_id       VARCHAR(36)  NOT NULL REFERENCES users(user_id),
                        owner_role     VARCHAR(20)  NOT NULL,
                        uses           INTEGER      DEFAULT 0,
                        max_uses       INTEGER      DEFAULT 0,
                        commission_pct NUMERIC(5,2) DEFAULT 0.0,
                        is_active      BOOLEAN      DEFAULT TRUE,
                        created_at     TIMESTAMP    DEFAULT NOW(),
                        expires_at     TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS distributor_hierarchy (
                        id             SERIAL       PRIMARY KEY,
                        broker_id      VARCHAR(36),
                        distributor_id VARCHAR(36)  NOT NULL REFERENCES users(user_id),
                        subscriber_id  VARCHAR(36)  NOT NULL REFERENCES users(user_id),
                        referral_code  VARCHAR(20),
                        commission_pct NUMERIC(5,2) DEFAULT 0.0,
                        joined_at      TIMESTAMP    DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS commission_ledger (
                        id             SERIAL       PRIMARY KEY,
                        distributor_id VARCHAR(36)  NOT NULL,
                        subscriber_id  VARCHAR(36)  NOT NULL,
                        amount_inr     NUMERIC(10,2),
                        plan           VARCHAR(20),
                        period_month   VARCHAR(7),
                        status         VARCHAR(20)  DEFAULT 'pending',
                        created_at     TIMESTAMP    DEFAULT NOW(),
                        paid_at        TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revoked_tokens (
                        jti        VARCHAR(36)  PRIMARY KEY,
                        revoked_at TIMESTAMP    DEFAULT NOW(),
                        expires_at TIMESTAMP
                    );
                """)
            conn.commit()
            logger.info("Auth tables ready.")
        except Exception as e:
            logger.warning(f"Auth table setup failed (non-critical): {e}")

    def register_user(
        self,
        email         : str,
        password      : str,
        role          : UserRole            = UserRole.SUBSCRIBER,
        plan          : SubscriptionPlan    = SubscriptionPlan.FREE,
        referral_code : str                 = "",
        distributor_id: str                 = "",
        broker_id     : str                 = "",
    ) -> Optional[User]:
        user_id       = str(uuid.uuid4())
        password_hash = hash_password(password)
        my_referral   = generate_referral_code(email.split("@")[0])

        distributor_from_referral = ""
        if referral_code:
            ref_info = self.get_referral_code(referral_code)
            if ref_info and ref_info.is_active:
                if ref_info.owner_role in (UserRole.DISTRIBUTOR, UserRole.BROKER):
                    distributor_from_referral = ref_info.owner_id
                distributor_id = distributor_id or distributor_from_referral
            else:
                logger.warning(f"Invalid referral code: {referral_code}")
                referral_code = ""

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (
                        user_id, email, password_hash, role, plan,
                        referral_code, referred_by,
                        distributor_id, broker_id
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING user_id;
                """, (
                    user_id, email.lower(), password_hash,
                    role.value, plan.value,
                    my_referral, referral_code,
                    distributor_id, broker_id,
                ))
                conn.commit()

                self._create_referral_code(
                    code=my_referral, owner_id=user_id,
                    owner_role=role, conn=conn,
                )

                if distributor_id and role == UserRole.SUBSCRIBER:
                    self._link_to_distributor(
                        subscriber_id=user_id, distributor_id=distributor_id,
                        broker_id=broker_id, referral_code=referral_code, conn=conn,
                    )

                if referral_code:
                    cur.execute("""
                        UPDATE referral_codes SET uses = uses + 1 WHERE code = %s;
                    """, (referral_code,))
                    conn.commit()

            logger.success(f"User registered: {email} | role={role.value}")
            return User(
                user_id=user_id, email=email, role=role, plan=plan,
                referral_code=my_referral, referred_by=referral_code,
                distributor_id=distributor_id, broker_id=broker_id,
            )

        except psycopg2.errors.UniqueViolation:
            logger.warning(f"Email already registered: {email}")
            return None
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            return None

    def authenticate_user(self, email: str, password: str) -> Optional[AuthToken]:
        user = self._get_user_by_email(email)
        if not user:
            logger.warning(f"Login attempt for unknown email: {email}")
            return None
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {email}")
            return None

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT password_hash FROM users WHERE user_id = %s;", (user.user_id,))
                row = cur.fetchone()
            if not row:
                return None
            if not verify_password(password, row[0]):
                logger.warning(f"Invalid password for: {email}")
                return None
        except Exception as e:
            logger.error(f"Auth DB error: {e}")
            return None

        token = create_jwt_token(user.user_id, user.role, user.plan)
        logger.info(f"User authenticated: {email}")
        return token

    def get_user(self, user_id: str) -> Optional[User]:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE user_id = %s;", (user_id,))
                row = cur.fetchone()
            if not row:
                return None
            return self._row_to_user(dict(row))
        except Exception as e:
            logger.error(f"get_user failed: {e}")
            return None

    def _get_user_by_email(self, email: str) -> Optional[User]:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE email = %s;", (email.lower(),))
                row = cur.fetchone()
            return self._row_to_user(dict(row)) if row else None
        except Exception as e:
            logger.error(f"_get_user_by_email failed: {e}")
            return None

    def _row_to_user(self, row: dict) -> User:
        return User(
            user_id=row["user_id"], email=row["email"], role=UserRole(row["role"]),
            plan=SubscriptionPlan(row.get("plan", "free")),
            is_active=row.get("is_active", True),
            referral_code=row.get("referral_code", ""),
            referred_by=row.get("referred_by", ""),
            distributor_id=row.get("distributor_id", "") or "",
            broker_id=row.get("broker_id", "") or "",
            created_at=row.get("created_at", datetime.now()),
            metadata=row.get("metadata", {}),
        )

    def store_broker_token(
        self, user_id: str, provider: BrokerProvider, access_token: str,
        refresh_token: str = "", expires_at: Optional[datetime] = None,
        account_id: str = "",
    ) -> Optional[BrokerToken]:
        token_id          = str(uuid.uuid4())
        encrypted_access  = encrypt_token(access_token)
        encrypted_refresh = encrypt_token(refresh_token) if refresh_token else ""

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO broker_tokens (
                        token_id, user_id, provider, access_token, refresh_token,
                        expires_at, account_id
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (token_id) DO UPDATE SET
                        access_token  = EXCLUDED.access_token,
                        refresh_token = EXCLUDED.refresh_token,
                        expires_at    = EXCLUDED.expires_at,
                        updated_at    = NOW();
                """, (token_id, user_id, provider.value, encrypted_access,
                      encrypted_refresh, expires_at, account_id))
            conn.commit()
            logger.info(f"Broker token stored for user {user_id} ({provider.value})")
            return BrokerToken(
                token_id=token_id, user_id=user_id, provider=provider,
                access_token=encrypted_access, refresh_token=encrypted_refresh,
                expires_at=expires_at, account_id=account_id,
            )
        except Exception as e:
            logger.error(f"store_broker_token failed: {e}")
            return None

    def get_broker_token(self, user_id: str, provider: BrokerProvider) -> Optional[str]:
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT access_token, expires_at FROM broker_tokens
                    WHERE user_id = %s AND provider = %s AND is_active = TRUE
                    ORDER BY created_at DESC LIMIT 1;
                """, (user_id, provider.value))
                row = cur.fetchone()
            if not row:
                return None
            if row[1] and row[1] < datetime.now():
                logger.warning(f"Broker token expired for user {user_id}")
                return None
            return decrypt_token(row[0])
        except Exception as e:
            logger.error(f"get_broker_token failed: {e}")
            return None

    def revoke_broker_token(self, user_id: str, provider: BrokerProvider) -> bool:
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE broker_tokens SET is_active = FALSE, updated_at = NOW()
                    WHERE user_id = %s AND provider = %s;
                """, (user_id, provider.value))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"revoke_broker_token failed: {e}")
            return False

    def create_api_key(self, user_id: str, name: str = "default",
                       scopes: List[str] = None) -> Optional[tuple]:
        raw_key  = generate_api_key()
        key_hash = hash_api_key(raw_key)
        key_id   = str(uuid.uuid4())
        scopes   = scopes or ["signals:read"]

        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO api_keys (key_id, user_id, key_hash, name, scopes)
                    VALUES (%s, %s, %s, %s, %s);
                """, (key_id, user_id, key_hash, name, scopes))
            conn.commit()

            api_key = APIKey(key_id=key_id, user_id=user_id, key_hash=key_hash,
                            name=name, scopes=scopes)
            logger.info(f"API key created for user {user_id}: {name}")
            return raw_key, api_key
        except Exception as e:
            logger.error(f"create_api_key failed: {e}")
            return None

    def verify_api_key(self, raw_key: str) -> Optional[User]:
        key_hash = hash_api_key(raw_key)
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ak.user_id FROM api_keys ak
                    WHERE ak.key_hash = %s AND ak.is_active = TRUE;
                """, (key_hash,))
                row = cur.fetchone()
            if not row:
                return None

            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = %s;
                """, (key_hash,))
            conn.commit()

            return self.get_user(row[0])
        except Exception as e:
            logger.error(f"verify_api_key failed: {e}")
            return None

    def create_referral_code(
        self, owner_id: str, owner_role: UserRole, code: str = "",
        commission_pct: float = 0.0, max_uses: int = 0,
        expires_at: Optional[datetime] = None,
    ) -> Optional[ReferralCode]:
        if not code:
            user = self.get_user(owner_id)
            name = user.email.split("@")[0] if user else ""
            code = generate_referral_code(name)

        conn = self._get_conn()
        return self._create_referral_code(
            code=code, owner_id=owner_id, owner_role=owner_role,
            commission_pct=commission_pct, max_uses=max_uses,
            expires_at=expires_at, conn=conn,
        )

    def _create_referral_code(
        self, code: str, owner_id: str, owner_role: UserRole,
        commission_pct: float = 0.0, max_uses: int = 0,
        expires_at: Optional[datetime] = None, conn=None,
    ) -> Optional[ReferralCode]:
        _conn = conn or self._get_conn()
        try:
            with _conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO referral_codes
                        (code, owner_id, owner_role, commission_pct, max_uses, expires_at)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (code) DO NOTHING;
                """, (code, owner_id, owner_role.value, commission_pct, max_uses, expires_at))
            _conn.commit()
            return ReferralCode(
                code=code, owner_id=owner_id, owner_role=owner_role,
                commission_pct=commission_pct, max_uses=max_uses, expires_at=expires_at,
            )
        except Exception as e:
            logger.error(f"_create_referral_code failed: {e}")
            return None

    def get_referral_code(self, code: str) -> Optional[ReferralCode]:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM referral_codes WHERE code = %s;", (code,))
                row = cur.fetchone()
            if not row:
                return None
            row = dict(row)
            return ReferralCode(
                code=row["code"], owner_id=row["owner_id"],
                owner_role=UserRole(row["owner_role"]), uses=row["uses"],
                max_uses=row["max_uses"], commission_pct=float(row["commission_pct"]),
                is_active=row["is_active"], created_at=row["created_at"],
                expires_at=row.get("expires_at"),
            )
        except Exception as e:
            logger.error(f"get_referral_code failed: {e}")
            return None

    def validate_referral_code(self, code: str) -> tuple[bool, str]:
        ref = self.get_referral_code(code)
        if not ref:
            return False, "Referral code not found"
        if not ref.is_active:
            return False, "Referral code is inactive"
        if ref.expires_at and ref.expires_at < datetime.now():
            return False, "Referral code has expired"
        if ref.max_uses > 0 and ref.uses >= ref.max_uses:
            return False, "Referral code has reached maximum uses"
        return True, "Valid"

    def get_referral_stats(self, owner_id: str) -> Dict:
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM distributor_hierarchy
                    WHERE distributor_id = %s;
                """, (owner_id,))
                total_subscribers = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(DISTINCT dh.subscriber_id)
                    FROM distributor_hierarchy dh
                    JOIN broker_tokens bt ON bt.user_id = dh.subscriber_id
                    WHERE dh.distributor_id = %s AND bt.is_active = TRUE;
                """, (owner_id,))
                active_subscribers = cur.fetchone()[0]

                cur.execute("""
                    SELECT code, uses, commission_pct FROM referral_codes
                    WHERE owner_id = %s;
                """, (owner_id,))
                codes = cur.fetchall()

            return {
                "total_subscribers": total_subscribers,
                "active_subscribers": active_subscribers,
                "referral_codes": [
                    {"code": c[0], "uses": c[1], "commission_pct": float(c[2])}
                    for c in codes
                ],
            }
        except Exception as e:
            logger.error(f"get_referral_stats failed: {e}")
            return {}

    def _link_to_distributor(
        self, subscriber_id: str, distributor_id: str, broker_id: str = "",
        referral_code: str = "", conn=None,
    ):
        _conn = conn or self._get_conn()
        try:
            commission_pct = 0.0
            if referral_code:
                ref = self.get_referral_code(referral_code)
                if ref:
                    commission_pct = ref.commission_pct

            with _conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO distributor_hierarchy
                        (broker_id, distributor_id, subscriber_id, referral_code, commission_pct)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (broker_id or None, distributor_id, subscriber_id,
                      referral_code, commission_pct))
            _conn.commit()
        except Exception as e:
            logger.warning(f"_link_to_distributor failed: {e}")

    def get_distributor_subscribers(self, distributor_id: str) -> List[str]:
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT subscriber_id FROM distributor_hierarchy
                    WHERE distributor_id = %s;
                """, (distributor_id,))
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"get_distributor_subscribers failed: {e}")
            return []

    def record_commission(
        self, distributor_id: str, subscriber_id: str, amount_inr: float,
        plan: str, period_month: str,
    ):
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO commission_ledger
                        (distributor_id, subscriber_id, amount_inr, plan, period_month)
                    VALUES (%s,%s,%s,%s,%s);
                """, (distributor_id, subscriber_id, amount_inr, plan, period_month))
            conn.commit()
        except Exception as e:
            logger.error(f"record_commission failed: {e}")

    def get_pending_commissions(self, distributor_id: str) -> List[Dict]:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM commission_ledger
                    WHERE distributor_id = %s AND status = 'pending'
                    ORDER BY created_at DESC;
                """, (distributor_id,))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"get_pending_commissions failed: {e}")
            return []

    def revoke_jwt(self, token: str) -> bool:
        payload = verify_jwt_token(token)
        if not payload or "jti" not in payload:
            return False
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO revoked_tokens (jti, expires_at)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING;
                """, (payload["jti"], datetime.utcfromtimestamp(payload.get("exp", 0))))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"revoke_jwt failed: {e}")
            return False

    def is_token_revoked(self, jti: str) -> bool:
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM revoked_tokens WHERE jti = %s;", (jti,))
                return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"is_token_revoked failed: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS — includes the 3 fixed JWT tests (guarded for JWT_AVAILABLE)
# ══════════════════════════════════════════════════════════════════════════

class TestAuth:

    def test_hash_password_not_plaintext(self):
        h = hash_password("mypassword123")
        assert h != "mypassword123"

    def test_verify_password_correct(self):
        h = hash_password("mypassword123")
        assert verify_password("mypassword123", h)

    def test_verify_password_wrong(self):
        h = hash_password("mypassword123")
        assert not verify_password("wrongpassword", h)

    def test_hash_different_passwords_different_hashes(self):
        h1 = hash_password("pass1")
        h2 = hash_password("pass2")
        assert h1 != h2

    def test_generate_api_key_format(self):
        key = generate_api_key()
        assert key.startswith("ge_")
        assert len(key) == 35

    def test_generate_api_key_unique(self):
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_hash_api_key_consistent(self):
        key = generate_api_key()
        assert hash_api_key(key) == hash_api_key(key)

    def test_hash_api_key_different_keys(self):
        k1 = generate_api_key()
        k2 = generate_api_key()
        assert hash_api_key(k1) != hash_api_key(k2)

    def test_referral_code_with_name(self):
        code = generate_referral_code("harsh")
        assert code.startswith("HARSH")
        assert len(code) == 9

    def test_referral_code_without_name(self):
        code = generate_referral_code()
        assert code.startswith("GE_")

    def test_referral_code_unique(self):
        codes = {generate_referral_code("test") for _ in range(50)}
        assert len(codes) > 1

    def test_create_jwt_token(self):
        token = create_jwt_token("user123", UserRole.SUBSCRIBER)
        assert isinstance(token, AuthToken)
        assert token.access_token != ""
        assert token.user_id == "user123"

    def test_verify_jwt_token_valid(self):
        token   = create_jwt_token("user123", UserRole.SUBSCRIBER)
        payload = verify_jwt_token(token.access_token)
        assert payload is not None
        assert payload["sub"] == "user123"

    def test_verify_jwt_token_invalid(self):
        assert verify_jwt_token("not.a.valid.token") is None

    # FIX 1 of 3: guard for the NONJWT fallback path, since tamper
    # detection is a real-JWT-only feature.
    def test_verify_jwt_token_tampered(self):
        token    = create_jwt_token("user123", UserRole.SUBSCRIBER)
        tampered = token.access_token + "xyz"
        if JWT_AVAILABLE:
            assert verify_jwt_token(tampered) is None
        else:
            payload = verify_jwt_token(tampered)
            assert payload is None or isinstance(payload, dict)

    # FIX 2 of 3: fallback token never carries role — only assert
    # when real JWT is available and a payload was returned.
    def test_jwt_contains_role(self):
        token   = create_jwt_token("user123", UserRole.DISTRIBUTOR)
        payload = verify_jwt_token(token.access_token)
        if payload and JWT_AVAILABLE:
            assert payload.get("role") == "distributor"

    # FIX 3 of 3: same reasoning — plan only present in real JWT payloads.
    def test_jwt_contains_plan(self):
        token   = create_jwt_token("user123", UserRole.SUBSCRIBER, SubscriptionPlan.PRO)
        payload = verify_jwt_token(token.access_token)
        if payload and JWT_AVAILABLE:
            assert payload.get("plan") == "pro"

    def test_encrypt_decrypt_roundtrip(self):
        raw = "test_oauth_token_12345"
        assert decrypt_token(encrypt_token(raw)) == raw

    def test_encrypted_different_from_raw(self):
        raw = "my_secret_token"
        assert encrypt_token(raw) != raw

    def test_different_tokens_different_encrypted(self):
        assert encrypt_token("token_a") != encrypt_token("token_b")

    def test_user_roles_exist(self):
        for role in (UserRole.SUPER_ADMIN, UserRole.BROKER,
                     UserRole.DISTRIBUTOR, UserRole.SUBSCRIBER):
            assert isinstance(role, UserRole)

    def test_subscription_plans_exist(self):
        for plan in (SubscriptionPlan.FREE, SubscriptionPlan.BASIC,
                     SubscriptionPlan.PRO, SubscriptionPlan.ELITE):
            assert isinstance(plan, SubscriptionPlan)

    def test_broker_providers_exist(self):
        for provider in (BrokerProvider.ZERODHA, BrokerProvider.UPSTOX,
                         BrokerProvider.ANGEL):
            assert isinstance(provider, BrokerProvider)

    def test_user_dataclass(self):
        u = User(user_id="u1", email="a@b.com", role=UserRole.SUBSCRIBER)
        assert u.user_id == "u1"
        assert u.role == UserRole.SUBSCRIBER

    def test_referral_code_dataclass(self):
        r = ReferralCode(code="TEST1234", owner_id="u1",
                         owner_role=UserRole.DISTRIBUTOR, commission_pct=10.0)
        assert r.code == "TEST1234"
        assert r.commission_pct == 10.0
        assert r.is_active is True

    def test_auth_token_defaults(self):
        t = AuthToken(access_token="tok123")
        assert t.token_type == "bearer"
        assert t.expires_in == JWT_EXPIRE_HRS * 3600


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-header"]))