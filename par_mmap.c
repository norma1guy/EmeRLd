#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
#define SIZE 256 * 1024 //EWRAM
#define SIZE2 32 * 1024 //IWRAM
#define SIZE3 240 * 160 //Pixels

typedef struct {
    uint32_t input;
    uint32_t size;
    char buffer[SIZE];
    char other_buffer[SIZE2];
    u_int32_t pixels[SIZE3];
    

} shm;

typedef struct {
    shm *ptr;
    sem_t *lua;
    sem_t *py;
} handler;


static int create_shm(lua_State *L){

    //Create flags for semaphores and shm
    const char *proc_num = lua_tostring(L,1);
    char lua_flag[64] = "/lua_to_py";
    char py_flag[64] = "/py_to_lua";
    char name[64] = "/RAM_MAP";

    strcat(lua_flag,proc_num);
    strcat(py_flag,proc_num);
    strcat(name,proc_num);


    //Release any old semaphores
    sem_unlink(lua_flag);
    sem_unlink(py_flag);
    shm_unlink(name);


    int shm_fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) return luaL_error(L,"shm_open failed");


    if(ftruncate(shm_fd,sizeof(shm)) == -1) {
        close(shm_fd);
        return luaL_error(L,"ftruncate failed");
    }

    shm *ptr = mmap(NULL,sizeof(shm),PROT_READ | PROT_WRITE, MAP_SHARED,shm_fd,0);
    close(shm_fd);
    if (ptr == MAP_FAILED) return luaL_error(L,"mmap failed");


    sem_t *lua_to_py = sem_open(lua_flag, O_CREAT, 0666,0);
    sem_t *py_to_lua = sem_open(py_flag, O_CREAT, 0666,1);
    if(lua_to_py == SEM_FAILED ){
        munmap(ptr, sizeof(shm));
        return luaL_error(L, "sem_open lua_to_py failed");
    }
    if(py_to_lua == SEM_FAILED) {
        munmap(ptr, sizeof(shm));
        sem_close(lua_to_py);
        return luaL_error(L, "sem_open failed");
    }

    handler *handle = lua_newuserdata(L, sizeof(handler));

    handle->ptr = ptr;
    handle->lua = lua_to_py;
    handle->py = py_to_lua;
    luaL_getmetatable(L, "shm_handler");
    lua_setmetatable(L, -2);

    return 1;
}

static int open_shm(lua_State *L){

    const char *proc_num = lua_tostring(L,1);
    char lua_flag[64] = "/lua_to_py";
    char py_flag[64] = "/py_to_lua";
    char name[64] = "/RAM_MAP";

    strcat(lua_flag,proc_num);
    strcat(py_flag,proc_num);
    strcat(name,proc_num);

    int shm_fd = shm_open(name,O_RDWR,0666);
    if (shm_fd == -1) return luaL_error(L,"shm_open failed");

    shm *ptr = mmap(NULL,sizeof(shm),PROT_READ | PROT_WRITE,MAP_SHARED, shm_fd,0);
    close(shm_fd);
    if(ptr == MAP_FAILED) return luaL_error(L, "mmap failed");
    sem_t *lua = sem_open(lua_flag,0);
    sem_t *py = sem_open(py_flag,0);
    if(lua == SEM_FAILED ){
        munmap(ptr, sizeof(shm));
        return luaL_error(L, "sem_open lua_to_py failed");
    }
    if(py == SEM_FAILED) {
        munmap(ptr, sizeof(shm));
        sem_close(lua);
        return luaL_error(L, "sem_open failed");
    }

    handler *handle = lua_newuserdata(L, sizeof(handler));
    handle->ptr = ptr;
    handle->lua = lua;
    handle->py = py;
    

    return 1;
}


static int handler_gc(lua_State *L) {
    handler *h = luaL_checkudata(L, 1, "shm_handler");
    if(h->ptr) munmap(h->ptr, sizeof(shm));
    if(h->lua) sem_close(h->lua);
    if(h->py) sem_close(h->py);
    return 0;
}

static int shm_write(lua_State *L) {
    handler *h = luaL_checkudata(L, 1, "shm_handler");
    //const char *msg = luaL_checkstring(L, 2);
    sem_wait(h->py);

    size_t len,len2,len3;
    const char *msg = luaL_checklstring(L,2,&len);
    const char *msg2 = luaL_checklstring(L,3,&len2);
    const char *msg3 = luaL_checklstring(L,4,&len3);

    if(len >= SIZE) len = SIZE;
    if(len2 >= SIZE2) len2 = SIZE2;
    if(len3 >= SIZE3) len3 = SIZE3;

    char *dest;
    memcpy(h->ptr->buffer, msg, len);
    memcpy(h->ptr->other_buffer, msg2, len2);
    memcpy(h->ptr->pixels,msg3,len3 * sizeof(uint32_t));

    h->ptr->size = len + len2 + len3;
    sem_post(h->lua);

    return 0;
}

static int shm_read(lua_State *L) {
    handler *h = luaL_checkudata(L, 1, "shm_handler");
    sem_wait(h->py);
    lua_pushnumber(L, h->ptr->input);
    sem_post(h->py);
    return 1;
}

int luaopen_par_shm(lua_State *L) {
    luaL_newmetatable(L, "shm_handler");
    lua_newtable(L);
    lua_pushcfunction(L,shm_write);
    lua_setfield(L,-2,"write");
    lua_pushcfunction(L,shm_read);
    lua_setfield(L,-2,"read");
    lua_setfield(L,-2,"__index");
    lua_pushcfunction(L, handler_gc);
    lua_setfield(L, -2, "__gc");
    lua_pop(L, 1);

    static const luaL_Reg funcs[] = {
        {"create_shm", create_shm},
        {"open_shm", open_shm},
        {NULL, NULL}
    };
    luaL_newlib(L, funcs);
    return 1;
}