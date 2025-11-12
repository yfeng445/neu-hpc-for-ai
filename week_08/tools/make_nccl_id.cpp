// tools/make_nccl_id.cpp
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(){
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  const unsigned char* p = reinterpret_cast<const unsigned char*>(&id);
  for (int i=0;i<128;i++) printf("%02x", (unsigned)p[i]);
  printf("\n");
  return 0;
}
