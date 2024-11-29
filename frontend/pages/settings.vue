<template>
    <div class="settings">
        <div class="setting-section">
            <h1>Postavke izgleda</h1>
            <div class="setting">
                <h2>Tema aplikacije</h2>
                <select class="theme-change" v-model="$colorMode.preference">
                    <option value="system">Zadano</option>
                    <option value="light">Svijetla</option>
                    <option value="dark">Tamna</option>
                </select>
            </div>
        </div>
        <div class="setting-section">
            <UButton to="/">
                LOGOUT
            </UButton>
        </div>
    </div>
</template>

<script setup lang="ts">

useHead({
    title: 'Postavke'
})

</script>


<style scoped>
.settings {
    padding: 0 5vw;
}

.setting-section {
    margin: 10vh 5vw;
}

.setting-section h1 {
    font-size: 1.5rem;
    font-weight: bold;
}

.setting {
    margin: 5vh 0;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
}

.theme-change {
    padding: 0.5rem;
    border-radius: 0.5rem;
    border: 1px solid #ccc;
}
</style>
