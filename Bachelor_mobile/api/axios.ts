import axios from "axios";
import Constants from "expo-constants";


const API_BASE_ADDRESS = Constants.expoConfig?.extra?.API_ADDRESS;

export const axiosInstance = axios.create({
    baseURL: API_BASE_ADDRESS,
})
