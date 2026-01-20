import redis

redis_client = redis.Redis(host='redis', port=6379, db=0)

SECONDS_IN_DAY = 86400
CACHE_EXPIRATION_SECONDS = SECONDS_IN_DAY * 2
